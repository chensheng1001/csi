import logging
from typing import Dict, Tuple

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn, optim as optim

import configs
import utils
from models.mine import create_tester


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
    
    logger.debug("Using Cross Entropy loss function for classifier.")
    criterion_clas = nn.CrossEntropyLoss()
    
    logger.debug("Using Cross Entropy loss function for discriminator.")
    criterion_disc = nn.CrossEntropyLoss()
    logger.debug("The weight for discriminator loss is {}.".format(loss_conf.lambda_disc))
    
    # optimizer
    logger.debug("Using Adam optimizer with default config")
    logger.debug("Using gradients reversal layer technique.")
    optimizer = optim.Adam(
            model.parameters())
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
        
        # set the module in training mode.
        model.train()
        
        # get net outputs
        labels_pred, _, domain_labels_pred, _, _ = model(
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
        ret.update(loss = loss_value)
        
        # backward propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return ret
    
    # construct engine
    trainer = utils.create_engine_with_logger(step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_total')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_clas']).attach(trainer, 'loss_classifier')
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss_disc']).attach(trainer, 'loss_discriminator')
    
    return trainer, (criterion_disc, criterion_clas), optimizers


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
    criterion_disc, criterion_clas = criteria
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
        
        # Sets the module in evaluation mode.
        model.eval()
        
        with torch.no_grad():
            # get net outputs
            labels_pred, labels_pred_soma, domain_labels_pred, domain_labels_pred_soma, _ = model(
                    samples, onehot_labels,
                    mode = 'whole')
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma,
                    domain_labels = domain_labels, domain_labels_pred = domain_labels_pred,
                    domain_labels_pred_softmaxed = domain_labels_pred_soma,
                    samples = samples)
    
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
    loss = ig_metrics.MetricsLambda(lambda a, b: a - loss_conf.lambda_disc * b, loss_clas, loss_disc)
    loss.attach(validator, 'loss_total')
    
    return validator


create_tester = create_tester
