import logging
from typing import Dict, Tuple

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn, optim as optim

import loss_function
import utils
from configs import Configuration
from models.ei import LossConfiguration


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
    loss_conf: LossConfiguration = conf.loss
    logger.debug("Using Cross Entropy loss function for classifier.")
    criterion_clas = nn.CrossEntropyLoss()
    logger.debug("Using Cross Entropy loss function for discriminator.")
    criterion_disc = nn.CrossEntropyLoss()
    logger.debug("The weight for discriminator loss is {}.".format(loss_conf.lambda_disc))
    logger.debug("Using Confidence Control Constraint.")
    confidence_constraint = loss_function.ConfidenceConstraint()
    logger.debug("The weight for Confidence Control Constraint is {}.".format(loss_conf.lambda_confidence))
    smoothness_constraint = loss_function.SmoothnessConstraint(model, loss_conf.smoothness_epsilon)
    logger.debug("The weight for Smoothing Constraint is {} with {}.".format(loss_conf.lambda_smoothness, loss_conf.smoothness_epsilon))
    # todo constraints
    
    # optimizer
    logger.debug("Using Adam optimizer with default parameters.")
    optimizer_base = optim.Adam(
            [{"params": model.feature_extractor.parameters()}, {"params": model.classifier.parameters()}])
    optimizer_disc = optim.Adam(
            model.domain_discriminators.parameters())
    optimizers = {'base': optimizer_base, 'discriminator': optimizer_disc}
    
    logger.debug("Using real adversarial training.")
    
    def adversarial_step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function, processing a batch of samples, with adversarial training.
        """
        # unpack batch
        samples, labels, _, domain_labels = batch
        samples = samples.to(device)
        labels = labels.to(device)
        domain_labels = domain_labels.to(device)
        
        # set the module in training mode.
        model.train()
        
        # fixme in the paper domain discriminator is updated after base network
        # update domain discriminator
        domain_labels_pred, _ = model(
                samples,
                mode = 'train_ad_domain')
        
        # calculate loss
        loss_disc = criterion_disc(domain_labels_pred, domain_labels)
        
        # backward propagate gradients
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
        
        # release memory
        del domain_labels_pred, _, loss_disc
        
        # update feature extractor and classifier
        labels_pred, _, domain_labels_pred, _, classifier_features = model(
                samples,
                mode = 'whole')
        
        # calculate loss
        loss_clas = criterion_clas(labels_pred, labels)
        loss_disc = criterion_disc(domain_labels_pred, domain_labels)
        loss = loss_clas - loss_conf.lambda_disc * loss_disc
        ret = dict(loss_disc = loss_disc.item(), loss_clas = loss_clas.item())
        
        loss_confidence = confidence_constraint(labels_pred, labels)
        loss += loss_conf.lambda_confidence * loss_confidence
        ret.update(loss_confidence = loss_confidence.item())
        
        loss_smoothness = smoothness_constraint(labels_pred, classifier_features)
        loss += loss_conf.lambda_smoothness * loss_smoothness
        ret.update(loss_smoothness = loss_smoothness.item())
        
        ret.update(loss = loss.item())
        
        # backward propagate gradients
        optimizer_base.zero_grad()
        loss.backward()
        optimizer_base.step()
        
        return ret
    
    # construct engine
    trainer = utils.create_engine_with_logger(adversarial_step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_total')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_clas']).attach(trainer, 'loss_classifier')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_disc']).attach(trainer, 'loss_discriminator')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_confidence']).attach(trainer, 'loss_confidence')
    
    return trainer, (criterion_disc, criterion_clas, confidence_constraint, smoothness_constraint), optimizers


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
    criterion_disc, criterion_clas, confidence_constraint, smoothness_constraint = criteria
    loss_conf: LossConfiguration = conf.loss
    
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
            labels_pred, labels_pred_soma, domain_labels_pred, domain_labels_pred_soma, classifier_features = model(
                    samples,
                    mode = 'whole')
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma,
                    domain_labels = domain_labels, domain_labels_pred = domain_labels_pred,
                    domain_labels_pred_softmaxed = domain_labels_pred_soma,
                    samples = samples, classifier_features = classifier_features)
    
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
    loss_confidence = ig_metrics.Loss(
            confidence_constraint,
            output_transform = lambda x: (x['labels_pred'], x['labels']),
            device = device)
    loss_confidence.attach(validator, 'loss_confidence')
    loss_smoothness = ig_metrics.Loss(
            smoothness_constraint,
            output_transform = lambda x: (x['labels_pred'], x['classifier_features']),
            device = device)
    loss_smoothness.attach(validator, 'loss_smoothness')
    
    def loss_cal(a, b, c, d):
        return a - loss_conf.lambda_disc * b + loss_conf.lambda_confidence * c + loss_conf.lambda_smoothness * d
    
    loss = ig_metrics.MetricsLambda(loss_cal, loss_clas, loss_disc, loss_confidence, loss_smoothness)
    loss.attach(validator, 'loss_total')
    
    return validator


def create_tester(
        name: str, model: nn.Module, device: torch.device,
        class_num: int) -> ig_engine.Engine:
    """
    Create a tester engine.

    :param name: Name of this engine.
    :param model: The model to test.
    :param device: The device that the model is using.
    :param class_num: The number of classes.
    :return: A tester engine.
    """
    _ = logging.getLogger(name)
    
    def inference(_: ig_engine.Engine, batch):
        """
        The main tester engine function, processing a batch of samples.
        """
        # unpack batch
        samples, labels, *_ = batch
        samples = samples.to(device)
        labels = labels.to(device)
        
        # Sets the module in evaluation mode.
        model.eval()
        
        # only classifier results is useful when testing
        with torch.no_grad():
            _, labels_pred_soma, _ = model(samples, mode = 'base')
        
        return dict(labels = labels, labels_pred_softmaxed = labels_pred_soma)
    
    # construct engine
    tester = utils.create_engine_with_logger(inference, name)
    
    # attach metrics for classification task
    _ = utils.attach_common_metrics(tester, utils.acc_out_transform, device)
    confusion_matrix = ig_metrics.ConfusionMatrix(
            num_classes = class_num,
            output_transform = utils.acc_out_transform,
            device = device)
    confusion_matrix.attach(tester, 'confusion_matrix')
    
    return tester
