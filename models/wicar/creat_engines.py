import functools
import logging
from typing import Callable, Dict, Tuple

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn, optim as optim

import utils
from configs import Configuration
from models.domain_adaptation_common.create_engines import create_tester


def create_trainer(
        name: str, model: nn.Module, device: torch.device,
        conf: Configuration) -> Tuple[ig_engine.Engine, Tuple[nn.Module, ...], Dict[str, optim.Optimizer]]:
    """
    Create trainer.
    
    :param name: Name of this engine.
    :param model: The model to train.
    :param device: The device that the model is using.
    :param conf: Configuration.
    :return: `(trainer engine, criteria, optimizers)`
    """
    logger = logging.getLogger(name)
    
    # loss functions
    loss_conf = conf.loss
    logger.debug("Using Cross Entropy loss function for classifier.")
    criterion_clas = nn.CrossEntropyLoss()
    logger.debug("Using Cross Entropy loss function for discriminator.")
    criterion_disc = nn.CrossEntropyLoss()
    logger.debug("The weight for discriminator loss is {}.".format(loss_conf.lambda_disc))
    
    # optimizer
    logger.debug("Using Adam optimizer with default parameters.")
    optimizer = optim.Adam(
            model.parameters())
    optimizers = {'optimizer': optimizer}
    
    logger.debug("Using gradients reversal layer technique.")
    
    def step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with gradients reversal layer technique.
        """
        # unpack batch
        samples, labels, _, *domain_labels_tuple = batch
        samples = samples.to(device)
        labels = labels.to(device)
        domain_labels_tuple = tuple(tensor.to(device) for tensor in domain_labels_tuple)
        
        # set the module in training mode.
        model.train()
        
        # get net outputs
        labels_pred, _, *domain_labels_pred_tuple = model(
                samples,
                mode = 'train_grl',
                lambda0 = loss_conf.lambda_disc)
        domain_labels_pred_tuple = tuple(domain_labels_pred_tuple)
        
        # calculate loss
        loss_clas = criterion_clas(labels_pred, labels)
        loss_disc_list = []
        assert len(domain_labels_pred_tuple) == len(domain_labels_tuple)
        for (domain_labels_pred, _), domain_labels in zip(domain_labels_pred_tuple, domain_labels_tuple):
            loss_disc_list.append(criterion_disc(domain_labels_pred, domain_labels))
        loss_disc = torch.sum(torch.stack(loss_disc_list))
        ret = dict(loss_disc = loss_disc.item(), loss_clas = loss_clas.item())
        # when using grl, weight for discriminator loss is in GradientReversalFunction
        loss = loss_clas + loss_disc
        loss_value = (loss_clas - loss_conf.lambda_disc * loss_disc).item()
        
        ret.update(loss = loss_value)
        
        # backward propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return ret
    
    trainer = utils.create_engine_with_logger(step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_total')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_clas']).attach(trainer, 'loss_classifier')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_disc']).attach(trainer, 'loss_discriminator')
    
    return trainer, (criterion_disc, criterion_clas), optimizers


def create_validator(
        name: str, model: nn.Module, device: torch.device,
        criteria: Tuple[nn.Module, ...],
        conf: Configuration) -> ig_engine.Engine:
    """
    
    The main difference between validator and tester is that loss are collected in validator.
    
    :param name: Name of this engine.
    :param model: The model to validate.
    :param device: The device that the model is using.
    :param criteria: The loss functions.
    :param conf: Configuration.
    :return: A validator engine.
    """
    _ = logging.getLogger(name)
    
    # loss functions
    criterion_disc, criterion_clas = criteria
    
    def inference(_: ig_engine.Engine, batch):
        """
        The main validator engine function, processing a batch of samples.
        """
        # unpack batch
        samples, labels, _, *domain_labels_tuple = batch
        samples = samples.to(device)
        labels = labels.to(device)
        domain_labels_tuple = tuple(tensor.to(device) for tensor in domain_labels_tuple)
        
        # Sets the module in evaluation mode.
        model.eval()
        
        with torch.no_grad():
            # get net outputs
            labels_pred, labels_pred_soma, *domain_labels_pred_tuple = model(
                    samples,
                    mode = 'whole')
            domain_labels_pred_tuple = tuple(domain_labels_pred_tuple)
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma,
                    domain_labels_tuple = domain_labels_tuple, domain_labels_pred_tuple = domain_labels_pred_tuple,
                    samples = samples)
    
    # construct engine
    validator = utils.create_engine_with_logger(inference, name)
    
    # attach metrics for classification task
    _ = utils.attach_common_metrics(validator, utils.acc_out_transform, device)
    
    # attach metrics for domain discrimination task
    def domain_acc_out_transform(output: dict, domain_category: int):
        """
        Transform the :class:`ignite.engine.Engine`'s `process_function`'s output into the form expected by metrics of
        domain discriminator.
        
        Should use softmaxed prediction to compute accuracy, precision, et c.
        
        :param output: The output of :class:`ignite.engine.Engine`'s `process_function`.
        :param domain_category: Domain category index.
        :return: y_pred, y
        """
        y_pred = output['domain_labels_pred_tuple'][domain_category][1]
        y = output['domain_labels_tuple'][domain_category]
        return y_pred, y
    
    for i in range(len(conf.network.domain_out_size)):
        _ = utils.attach_common_metrics(validator, functools.partial(domain_acc_out_transform, domain_category = i),
                                        device, 'domain{}'.format(i))
    
    # average loss
    def sum_loss(labels_pred_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                 labels_tuple: Tuple[torch.Tensor, ...],
                 loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Apply the same loss function to multiple label categories, and return the sum of loss.
        
        :param labels_pred_tuple:
        :param labels_tuple:
        :param loss_function:
        :return:
        """
        loss_list = []
        for (domain_labels_pred, _), domain_labels in zip(labels_pred_tuple, labels_tuple):
            loss_list.append(loss_function(domain_labels_pred, domain_labels))
        loss_sum = torch.sum(torch.stack(loss_list))
        return loss_sum
    
    # losses are calculated when evaluating the model for validation purpose
    loss_clas = ig_metrics.Loss(
            criterion_clas,
            output_transform = lambda x: (x['labels_pred'], x['labels']),
            device = device)
    loss_clas.attach(validator, 'loss_classifier')
    loss_disc = ig_metrics.Loss(
            functools.partial(sum_loss, loss_function = criterion_disc),
            output_transform = lambda x: (x['domain_labels_pred_tuple'], x['domain_labels_tuple']),
            batch_size = lambda x_tuple: len(x_tuple[0]),
            device = device)
    loss_disc.attach(validator, 'loss_discriminator')
    loss = ig_metrics.MetricsLambda(lambda a, b: a - conf.loss.lambda_disc * b, loss_clas, loss_disc)
    loss.attach(validator, 'loss_total')
    
    return validator


create_tester = create_tester
