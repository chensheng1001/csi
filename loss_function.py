from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as func
from torch.nn.modules.loss import _Loss as _Loss, _WeightedLoss as _WeightedLoss


def smoothen_label(labels: torch.Tensor, smooth_factor: float, class_num: int) -> torch.Tensor:
    """
    Apply label smoothening to the one-hot labels.
    
    :param labels: One-hot labels, must of type floats.
    :param smooth_factor: Label smooth factor, should be between 0.0 and 1.0.
    :param class_num: Number of classes.
    :return: Smoothened One-hot labels.
    """
    assert 0.0 <= smooth_factor < 1.0
    
    with torch.no_grad():
        labels_fake = labels.new_empty(labels.size())
        labels_fake.fill_(smooth_factor / (class_num - 1))
        
        labels_softened = torch.mul(labels, 1 - smooth_factor)
        
        labels_smoothened = torch.where(labels_softened > 0.0, labels_softened, labels_fake)
        labels_smoothened = labels_smoothened.detach()  # have no effect since already under no_grad context
    return labels_smoothened


class LabelSmoothingLoss(_WeightedLoss):
    """
    Cross-entropy loss with label smoothing.
    
    
    :param class_num: Number of classes.
    :param smoothing: Label smoothing factor.
    :param weight: A manual rescaling weight given to each class.If given, has to be a Tensor of size `C`.
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    
    __constants__ = ['class_num', 'smoothing', 'confidence', 'weight', 'reduction']
    
    def __init__(self, class_num: int, smoothing: float = 0.1, weight: torch.Tensor = None, reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__(weight = weight, reduction = reduction)
        
        self.class_num = class_num
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        :param pred: Predicted logits, of shape `(N, C, ...)`.
        :param target: True labels, of shape `(N, ...)`.
        :return: Loss.
        """
        with torch.no_grad():
            target_onehot = pred.new_empty(pred.size())
            target_onehot.fill_(self.smoothing / (self.class_num - 1))
            target_onehot.scatter_(1, target.unsqueeze(1), self.confidence)
            target_onehot = target_onehot.detach()  # have no effect since already under no_grad context

        log_prob = func.log_softmax(pred, dim = 1)
        if self.weight is not None:
            log_prob = log_prob * self.weight.unsqueeze(0)
        
        loss = torch.sum(torch.mul(target_onehot, log_prob), dim = 1).neg()
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss


class CosineSimilarityLoss(_Loss):
    r"""Creates a criterion that measures the loss given input labels :math:`y_{pred}`, :math:`y`.This is used for
    measuring the similarity between two inputs using the cosine distance.
    
    The loss function for each sample is:
    
    .. math::
        \text{loss}(y_{pred}, y) = 1 - \cos(y_{pred}, \text{onehot}(y))
    
    :param class_num: Number of classes.
    :param from_logits: If true, then :math:`y_{pred}` should be logits and will go through softmax function.
    :param smoothing: Label smoothing factor. If provided, will smooth :math:`y` before computing loss.
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    __constants__ = ['class_num', 'smoothing', 'reduction']
    
    def __init__(self, class_num: int, from_logits: bool = False, smoothing: Optional[float] = None,
                 reduction: str = 'mean'):
        super(CosineSimilarityLoss, self).__init__(reduction = reduction)
        
        self.class_num = class_num
        self.from_logits = from_logits
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        :param pred: Predicted labels of logits (if from_logits is true), of shape `(N, C)`.
        :param target: True labels, of shape `(N)`.
        :return: Loss.
        """
        if self.from_logits:
            pred = func.softmax(pred, dim = 1)
        
        target_onehot = func.one_hot(target, self.class_num).to(pred.dtype)
        
        if self.smoothing is not None:
            target_onehot = smoothen_label(target_onehot, self.smoothing, self.class_num)
        
        dim = pred.shape[0]
        return func.cosine_embedding_loss(pred, target_onehot, pred.new_ones(dim),
                                          margin = 0, reduction = self.reduction)


class SmoothnessRegularizer(_Loss):
    """
    Feature space smoothness regularization.
    
    Similar to CsiGAN's manifold regularization. Similar regularization are also used in many semi-supervised learning.
    
    :param model: The model that has feature_extractor and classifier sub-module.
    :param eta: eta.
    :param alpha: alpha.
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    
    __constants__ = ['eta', 'alpha', 'reduction']
    
    def __init__(self, model: nn.Module, eta: float, alpha: float, reduction: str = 'mean'):
        super(SmoothnessRegularizer, self).__init__(reduction = reduction)
        
        self.model = model
        self.eta = eta
        self.alpha = alpha
    
    def forward(self, labels_pred: torch.Tensor, samples: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        :param labels_pred: The batch of label predictions by classifier.
        :param samples: The batch of samples.
        :param features: The batch of features.
        :return: the feature space smoothness regularization loss.
        """
        labels_pred = labels_pred.detach()
        
        samples_perturbed = samples + self.eta * torch.randn_like(samples)
        features_perturbed = self.model.feature_extractor(samples_perturbed)
        
        if features is None:
            features = self.model.feature_extractor(samples)
        features_perturbed = torch.add(torch.mul(features.detach(), 1 - self.alpha),
                                       torch.mul(features_perturbed, self.alpha))
        
        labels_pred_perturbed, _ = self.model.classifier(features_perturbed)
        
        return func.l1_loss(labels_pred_perturbed, labels_pred, reduction = self.reduction)


class ConfidenceConstraint(_WeightedLoss):
    """
     Confidence control constraint from EI.
    
    :param weight: A manual rescaling weight given to each class.If given, has to be a Tensor of size `C`.
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    
    __constants__ = ['weight', 'reduction']
    
    def __init__(self, weight: torch.Tensor = None, reduction: str = 'mean'):
        super(ConfidenceConstraint, self).__init__(weight = weight, reduction = reduction)
    
    def forward(self, pred: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        :param pred: Predicted logits, of shape `(N, C, ...)`.
        :param _: True labels, of shape `(N, ...)`.
        :return: Loss.
        """
        
        log_prob = func.log_softmax(pred, dim = 1)
        if self.weight is not None:
            log_prob = log_prob * self.weight.unsqueeze(0)
        
        log_prob2 = torch.add(func.softmax(pred, dim = 1).neg(), 1).log()
        # fixme While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower,
        #  and numerically unstable.
        if self.weight is not None:
            log_prob2 = log_prob2 * self.weight.unsqueeze(0)
        
        loss = torch.sum(torch.add(log_prob, log_prob2), dim = 1).neg()
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss


class SmoothnessConstraint(_Loss):
    """
    Feature space smoothness constraint from EI.
    
    :param model: The model that has feature_extractor and classifier sub-module.
    :param epsilon: epsilon.
    :param reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    
    __constants__ = ['epsilon', 'reduction']
    
    def __init__(self, model: nn.Module, epsilon: float, reduction: str = 'mean'):
        super(SmoothnessConstraint, self).__init__(reduction = reduction)
        
        self.model = model
        self.epsilon = epsilon
    
    def forward(self, labels_pred: torch.Tensor, classifier_features: torch.Tensor = None) -> torch.Tensor:
        """
        :param labels_pred: The batch of label predictions by classifier.
        :param classifier_features: The batch of classifier features.
        :return: the feature space smoothness regularization loss.
        """
        labels_pred = labels_pred.detach()
        
        features_perturbed = classifier_features + self.epsilon * torch.randn_like(classifier_features)
        
        labels_pred_perturbed = self.model.classifier.softmax_layer(features_perturbed)
        
        p_output = func.softmax(labels_pred_perturbed, dim = 1)
        q_output = func.softmax(labels_pred, dim = 1)
        log_mean_output = ((p_output + q_output) / 2).log()
        reduction = 'batchmean' if self.reduction == 'mean' else self.reduction
        
        return (func.kl_div(log_mean_output, p_output, reduction = reduction) +
                func.kl_div(log_mean_output, q_output, reduction = reduction)) / 2
