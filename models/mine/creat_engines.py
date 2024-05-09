import logging
from typing import Dict, Tuple

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn, optim as optim

import configs
import loss_function
import utils
from models.domain_adaptation_common.create_engines import create_tester
from prediction_gan import PredictionStep


def create_trainer(
        name: str, model: nn.Module, device: torch.device,
        conf: configs.Configuration) -> Tuple[ig_engine.Engine, Tuple[nn.Module, ...], Dict[str, optim.Optimizer]]:
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
    
    if loss_conf.classifier_type == 'cos':
        logger.debug("Using Cosine Similarity loss function for classifier.")
        if loss_conf.classifier_label_smoothen:
            criterion_clas = loss_function.CosineSimilarityLoss(
                    conf.class_num,
                    from_logits = True,
                    smoothing = loss_conf.classifier_label_smooth_factor)
        else:
            criterion_clas = loss_function.CosineSimilarityLoss(
                    conf.class_num,
                    from_logits = True)
    elif loss_conf.classifier_type == 'cross_entropy':
        logger.debug("Using Cross Entropy loss function for classifier.")
        if loss_conf.classifier_label_smoothen:
            criterion_clas = loss_function.LabelSmoothingLoss(
                    conf.class_num,
                    loss_conf.classifier_label_smooth_factor)
        else:
            criterion_clas = nn.CrossEntropyLoss()
    else:
        message = "Unknown loss function {} set for classifier.".format(loss_conf.classifier_type)
        logger.error(message)
        raise ValueError(message)
    if loss_conf.classifier_label_smoothen:
        # When Does Label Smoothing Help?
        logger.debug("Using label smoothing {} for classifier.".format(loss_conf.classifier_label_smooth_factor))
    
    logger.debug("Using Cross Entropy loss function for discriminator.")
    if loss_conf.discriminator_label_smoothen:
        logger.debug("Using label smoothing {} for discriminator.".format(loss_conf.discriminator_label_smooth_factor))
        criterion_disc = loss_function.LabelSmoothingLoss(
                conf.class_num,
                loss_conf.discriminator_label_smooth_factor)
    else:
        criterion_disc = nn.CrossEntropyLoss()
    logger.debug("The weight for discriminator loss is {}.".format(loss_conf.lambda_disc))
    
    # feature space smoothness regularization
    if loss_conf.regularization_smooth:
        logger.debug("Using feature space smoothness regularization.")
        logger.debug(
                "The weight for feature space smoothness regularization loss is {}.".format(
                        loss_conf.lambda_smooth))
        logger.debug("Feature space smoothness regularizer's eta is {}, alpha is {}.".format(loss_conf.smooth_eta,
                                                                                             loss_conf.smooth_alpha))
        regularizer_smooth = loss_function.SmoothnessRegularizer(
                model,
                eta = loss_conf.smooth_eta,
                alpha = loss_conf.smooth_alpha)
    else:
        regularizer_smooth = None
    # todo Wasserstein distance
    # todo schedule lambdas in loss
    
    # optimizer
    optim_conf = conf.optimizer
    # todo learning rate schedule? if use, might should also be saved together with model.
    # todo SGD with momentum
    logger.debug(
            "Using Adam optimizer, learning rate is {}, beta is ({}, {}), epsilon is {}".format(
                    optim_conf.learning_rate,
                    optim_conf.beta_1,
                    optim_conf.beta_2,
                    optim_conf.eps))
    if conf.adversarial_training:
        logger.debug("Using real adversarial training.")
        # adversarial training optimizers
        optimizer_base = optim.Adam(
                [{"params": model.feature_extractor.parameters()}, {"params": model.classifier.parameters()}],
                lr = optim_conf.learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        if optim_conf.t_t_u_r:
            # use a separate learning rate, GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
            # Equilibrium
            logger.debug(
                    "Use a separate learning rate for discriminator as TTUR suggests. Learning rate is {}.".format(
                            optim_conf.learning_rate_discriminator))
            discriminator_learning_rate = optim_conf.learning_rate_discriminator
        else:
            discriminator_learning_rate = optim_conf.learning_rate
        optimizer_disc = optim.Adam(
                model.domain_discriminators.parameters(),
                lr = discriminator_learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        del discriminator_learning_rate
        
        # Stabilizing Adversarial Nets With Prediction Methods
        optimizer_pred_base = PredictionStep(
                [{"params": model.feature_extractor.parameters()}, {"params": model.classifier.parameters()}])
        optimizer_pred_disc = PredictionStep(
                model.domain_discriminators.parameters())
        optimizers = {
                'base': optimizer_base,
                'discriminator': optimizer_disc,
                'base_prediction_step': optimizer_pred_base,
                'discriminator_prediction_step': optimizer_pred_disc}
    else:
        logger.debug("Using gradients reversal layer technique.")
        optimizer = optim.Adam(
                model.parameters(),
                lr = optim_conf.learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        optimizers = {'optimizer': optimizer}
    
    def step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with gradients reversal layer technique.
        """
        # unpack batch
        samples, labels, onehot_labels, domain_labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        onehot_labels = onehot_labels.to(device)
        domain_labels = domain_labels.to(device)

        # apply label smoothening
        if loss_conf.classifier_label_smoothen:
            onehot_labels = loss_function.smoothen_label(onehot_labels.to(samples.dtype),
                                                         loss_conf.classifier_label_smooth_factor, conf.class_num)
        
        # set the module in training mode.
        model.train()
        
        # get net outputs
        labels_pred, _, domain_labels_pred, _, features = model(
                samples, onehot_labels,
                mode = 'train_grl',
                lambda0 = loss_conf.lambda_disc)
        
        # calculate loss
        loss_clas = criterion_clas(labels_pred, labels)
        loss_disc = criterion_disc(domain_labels_pred, domain_labels)
        ret = dict(loss_disc = loss_disc.item(), loss_clas = loss_clas.item())
        # when using grl, weight for discriminator loss is in GradientReversalFunction
        loss = loss_clas + loss_disc
        loss_value = (loss_clas - loss_conf.lambda_disc * loss_disc).item()
        
        if loss_conf.regularization_smooth:
            # feature space smoothness regularization
            loss_smooth = regularizer_smooth(labels_pred, samples, features)
            loss += loss_conf.lambda_smooth * loss_smooth
            loss_value += (loss_conf.lambda_smooth * loss_smooth).item()
            ret.update(loss_smooth = loss_smooth.item())
        
        ret.update(loss = loss_value)
        
        # backward propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return ret
    
    def adversarial_step(engine: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with adversarial training.
        """
        # unpack batch
        samples, labels, onehot_labels, domain_labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        onehot_labels = onehot_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        # apply label smoothening
        if loss_conf.classifier_label_smoothen:
            onehot_labels = loss_function.smoothen_label(onehot_labels.to(samples.dtype),
                                                         loss_conf.classifier_label_smooth_factor, conf.class_num)
        
        # set the module in training mode.
        model.train()
        
        if conf.stable_adversarial_training:
            lookahead_step = 1.0
        else:
            # no prediction step
            lookahead_step = 0.0
        
        # avoid lookahead during the first iteration
        if engine.state.iteration == 1:
            lookahead_step = 0.0
        
        # update domain discriminator
        with optimizer_pred_base.lookahead(step = lookahead_step):
            domain_labels_pred, _ = model(
                    samples, onehot_labels,
                    mode = 'train_ad_domain')
            
            # calculate loss
            loss_disc = criterion_disc(domain_labels_pred, domain_labels)
            
            # backward propagate gradients
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()
            optimizer_pred_disc.step()
            
            # release memory
            del domain_labels_pred, _, loss_disc
        
        # update feature extractor and classifier
        with optimizer_pred_disc.lookahead(step = lookahead_step):
            labels_pred, _, domain_labels_pred, _, features = model(
                    samples, onehot_labels,
                    mode = 'whole')
        
            # calculate loss
            loss_clas = criterion_clas(labels_pred, labels)
            loss_disc = criterion_disc(domain_labels_pred, domain_labels)
            ret = dict(loss_disc = loss_disc.item(), loss_clas = loss_clas.item())
            loss = loss_clas - loss_conf.lambda_disc * loss_disc
            
            if loss_conf.regularization_smooth:
                # feature space smoothness regularization
                loss_smooth = regularizer_smooth(labels_pred, samples, features)
                loss += loss_conf.lambda_smooth * loss_smooth
                ret.update(loss_smooth = loss_smooth.item())
            
            ret.update(loss = loss.item())
            
            # backward propagate gradients
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()
            optimizer_pred_base.step()
        
        return ret
    
    # construct engine
    if conf.adversarial_training:
        trainer = utils.create_engine_with_logger(adversarial_step, name)
    else:
        trainer = utils.create_engine_with_logger(step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_total')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_clas']).attach(trainer, 'loss_classifier')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_disc']).attach(trainer, 'loss_discriminator')
    if loss_conf.regularization_smooth:
        ig_metrics.RunningAverage(output_transform = lambda x: x['loss_smooth']).attach(trainer, 'loss_smoothness')
    
    return trainer, (criterion_disc, criterion_clas, regularizer_smooth), optimizers


def create_validator(
        name: str, model: nn.Module, device: torch.device,
        criteria: Tuple[nn.Module, ...],
        conf: configs.Configuration) -> ig_engine.Engine:
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
    criterion_disc, criterion_clas, regularizer_smooth = criteria
    loss_conf = conf.loss
    
    def inference(_: ig_engine.Engine, batch):
        """
        The main validator engine function, processing a batch of samples.
        """
        # unpack batch
        samples, labels, onehot_labels, domain_labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        onehot_labels = onehot_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        # apply label smoothening
        if loss_conf.classifier_label_smoothen:
            onehot_labels = loss_function.smoothen_label(onehot_labels.to(samples.dtype),
                                                         loss_conf.classifier_label_smooth_factor, conf.class_num)
        
        # Sets the module in evaluation mode.
        model.eval()
        
        with torch.no_grad():
            # get net outputs
            labels_pred, labels_pred_soma, domain_labels_pred, domain_labels_pred_soma, features = model(
                    samples, onehot_labels,
                    mode = 'whole')
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma,
                    domain_labels = domain_labels, domain_labels_pred = domain_labels_pred,
                    domain_labels_pred_softmaxed = domain_labels_pred_soma,
                    samples = samples, features = features)
    
    # construct engine
    validator = utils.create_engine_with_logger(inference, name)
    
    # attach metrics for classification task
    _ = utils.attach_common_metrics(validator, utils.acc_out_transform, device)
    # attach metrics for domain discrimination task
    _ = utils.attach_common_metrics(validator, utils.domain_acc_out_transform, device, 'domain')
    # average loss
    # losses are calculated when evaluating the model for validation purpose
    loss_clas = ig_metrics.Loss(
            criterion_clas,
            output_transform = lambda x: (x['labels_pred'], x['labels']),
            device = device)
    loss_clas.attach(validator, 'loss_classifier')
    loss_disc = ig_metrics.Loss(
            criterion_disc,
            output_transform = lambda x: (x['domain_labels_pred'], x['domain_labels']),
            device = device)
    loss_disc.attach(validator, 'loss_discriminator')
    # feature space smoothness regularization
    if loss_conf.regularization_smooth:
        loss_smooth = ig_metrics.Loss(
                regularizer_smooth,
                output_transform = lambda x: (x['labels_pred'], x['samples'], dict(features = x['features'])),
                device = device)
        loss_smooth.attach(validator, 'loss_smoothness')
        
        loss = ig_metrics.MetricsLambda(lambda a, b, c: a - loss_conf.lambda_disc * b + loss_conf.lambda_smooth * c,
                                        loss_clas, loss_disc, loss_smooth)
    else:
        loss = ig_metrics.MetricsLambda(lambda a, b: a - loss_conf.lambda_disc * b, loss_clas, loss_disc)
    loss.attach(validator, 'loss_total')
    
    return validator


def create_fine_tune_trainer(
        name: str, model: nn.Module, device: torch.device,
        conf: configs.Configuration) -> Tuple[ig_engine.Engine, Tuple[nn.Module, ...], Dict[str, optim.Optimizer]]:
    """
    Create trainer that uses unlabeled data to fine-tune model in the target domain.
    
    :param name: Name of this engine.
    :param model: The model to train.
    :param device: The device that the model is using.
    :param conf: Configuration.
    :return: `(trainer engine, criteria, optimizers)`
    """
    logger = logging.getLogger(name)
    
    logger.info("Using {} domain feature.".format(conf.network.domain_feature_type))
    
    # loss functions
    loss_conf = conf.loss
    
    logger.info("Using feature space smoothness regularization.")
    logger.debug(
            "Feature space smoothness regularizer's eta is {}, alpha is {}.".format(loss_conf.smooth_eta,
                                                                                    loss_conf.smooth_alpha))
    regularizer_smooth = loss_function.SmoothnessRegularizer(model,
                                                             eta = loss_conf.smooth_eta,
                                                             alpha = loss_conf.smooth_alpha)
    
    logger.debug("Using Cross Entropy loss function for discriminator.")
    if loss_conf.discriminator_label_smoothen:
        logger.debug("Using label smoothing {} for discriminator.".format(loss_conf.discriminator_label_smooth_factor))
        criterion_disc = loss_function.LabelSmoothingLoss(
                conf.class_num,
                loss_conf.discriminator_label_smooth_factor)
    else:
        criterion_disc = nn.CrossEntropyLoss()
    logger.debug("The weight for discriminator loss is {}.".format(loss_conf.lambda_disc_fine_tune))
    
    # optimizer
    optim_conf = conf.optimizer
    logger.debug("Using Adam optimizer, learning rate is {}, beta is ({}, {})".format(optim_conf.learning_rate,
                                                                                      optim_conf.beta_1,
                                                                                      optim_conf.beta_2))
    if conf.adversarial_training:
        logger.debug("Using real adversarial training.")
        # adversarial training optimizers
        optimizer_base = optim.Adam(
                [{"params": model.feature_extractor.parameters()}, {"params": model.classifier.parameters()}],
                lr = optim_conf.learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        if optim_conf.t_t_u_r:
            # use a separate learning rate, GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
            # Equilibrium
            logger.debug(
                    "Use a separate learning rate for discriminator as TTUR suggests. Learning rate is {}.".format(
                            optim_conf.learning_rate_discriminator))
            discriminator_learning_rate = optim_conf.learning_rate_discriminator
        else:
            discriminator_learning_rate = optim_conf.learning_rate
        optimizer_disc = optim.Adam(
                model.domain_discriminators.parameters(),
                lr = discriminator_learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        del discriminator_learning_rate
        
        # Stabilizing Adversarial Nets With Prediction Methods
        optimizer_pred_base = PredictionStep(
                [{"params": model.feature_extractor.parameters()}, {"params": model.classifier.parameters()}])
        optimizer_pred_disc = PredictionStep(
                model.domain_discriminators.parameters())
        optimizers = {
                'base': optimizer_base,
                'discriminator': optimizer_disc,
                'base_prediction_step': optimizer_pred_base,
                'discriminator_prediction_step': optimizer_pred_disc}
    else:
        logger.debug("Using gradients reversal layer technique.")
        optimizer = optim.Adam(
                model.parameters(),
                lr = optim_conf.learning_rate,
                betas = (optim_conf.beta_1, optim_conf.beta_2),
                eps = optim_conf.eps)
        optimizers = {'optimizer': optimizer}
    
    def step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with gradients reversal layer technique.
        """
        # unpack batch
        samples, _, _, domain_labels = batch
        samples = samples.to(device)
        domain_labels = domain_labels.to(device)
        
        # set the module in training mode.
        model.train()
        
        # get net outputs
        labels_pred, _, domain_labels_pred, domain_labels_pred_soma, features = model(
                samples,
                mode = 'train_grl',
                lambda0 = loss_conf.lambda_disc_fine_tune)
        
        # calculate loss
        # feature space smoothness regularization
        loss_smooth = regularizer_smooth(labels_pred, samples, features)
        loss_disc = criterion_disc(domain_labels_pred, domain_labels)
        # when using grl, weight for discriminator loss is in GradientReversalFunction
        loss = loss_smooth + loss_disc
        loss_value = (loss_smooth - loss_conf.lambda_disc * loss_disc).item()
        
        # backward propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return dict(loss = loss_value, loss_disc = loss_disc.item(), loss_smooth = loss_smooth.item())
    
    def adversarial_step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with adversarial training.
        """
        # unpack batch
        samples, _, _, domain_labels = batch
        samples = samples.to(device)
        domain_labels = domain_labels.to(device)
        
        # set the module in training mode.
        model.train()
        
        if conf.stable_adversarial_training:
            lookahead_step = 1.0
        else:
            # no prediction step
            lookahead_step = 0.0
        
        # update domain discriminator
        with optimizer_pred_base.lookahead(step = lookahead_step):
            domain_labels_pred, _ = model(samples, mode = 'train_ad_domain')
            
            # calculate loss
            loss_disc = criterion_disc(domain_labels_pred, domain_labels)
            
            # backward propagate gradients
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()
            optimizer_pred_disc.step()
            
            # release memory
            del domain_labels_pred, loss_disc, _
        
        # update feature extractor and classifier
        with optimizer_pred_disc.lookahead(step = lookahead_step):
            labels_pred, _, domain_labels_pred, _, features = model(samples, mode = 'whole')
            
            # calculate loss
            # feature space smoothness regularization
            loss_smooth = regularizer_smooth(labels_pred, samples, features)
            loss_disc = criterion_disc(domain_labels_pred, domain_labels)
            loss = loss_smooth - loss_conf.lambda_disc_fine_tune * loss_disc
            
            # backward propagate gradients
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()
            optimizer_pred_base.step()
        
        return dict(loss = loss.item(), loss_disc = loss_disc.item(), loss_smooth = loss_smooth.item())
    
    # construct engine
    if conf.adversarial_training:
        trainer = utils.create_engine_with_logger(adversarial_step, name)
    else:
        trainer = utils.create_engine_with_logger(step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_total')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_smooth']).attach(trainer, 'loss_smoothness')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_disc']).attach(trainer, 'loss_discriminator')
    
    return trainer, (regularizer_smooth, criterion_disc), optimizers


def create_fine_tune_validator(
        name: str, model: nn.Module, device: torch.device,
        criteria: Tuple[nn.Module, ...],
        conf: configs.Configuration) -> ig_engine.Engine:
    """
    Create validator that works in fine-tune phase, evaluating with unlabeled data to monitor training.
    
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
    regularizer_smooth, criterion_disc = criteria
    loss_conf = conf.loss
    
    def inference(_: ig_engine.Engine, batch):
        """
        The main validator engine function, processing a batch of samples.
        """
        # unpack batch
        samples, labels, _, domain_labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        domain_labels = domain_labels.to(device)
        
        # Sets the module in evaluation mode.
        model.eval()
        
        with torch.no_grad():
            # get net outputs
            labels_pred, labels_pred_soma, domain_labels_pred, domain_labels_pred_soma, features = model(
                    samples,
                    mode = 'whole')
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma,
                    domain_labels = domain_labels, domain_labels_pred = domain_labels_pred,
                    domain_labels_pred_softmaxed = domain_labels_pred_soma,
                    samples = samples, features = features)
    
    # construct engine
    validator = utils.create_engine_with_logger(inference, name)
    
    # attach metrics for classification task
    _ = utils.attach_common_metrics(validator, utils.acc_out_transform, device)
    # attach metrics for domain discrimination task
    _ = utils.attach_common_metrics(validator, utils.domain_acc_out_transform, device, 'domain')
    # average loss
    # feature space smoothness regularization
    loss_smooth = ig_metrics.Loss(
            regularizer_smooth,
            output_transform = lambda x: (x['labels_pred'], x['samples'], dict(features = x['features'])),
            device = device)
    loss_smooth.attach(validator, 'loss_smoothness')
    loss_disc = ig_metrics.Loss(
            criterion_disc,
            output_transform = lambda x: (x['domain_labels_pred'], x['domain_labels']),
            device = device)
    loss_disc.attach(validator, 'loss_discriminator')
    loss = ig_metrics.MetricsLambda(lambda a, b: a - loss_conf.lambda_disc_fine_tune * b, loss_smooth, loss_disc)
    loss.attach(validator, 'loss_total')
    
    return validator


create_tester = create_tester
