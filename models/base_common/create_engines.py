import logging
from typing import Dict, Tuple

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn, optim as optim

import configs
import utils


def create_trainer(
        name: str, model: nn.Module, device: torch.device,
        _: configs.Configuration) -> Tuple[ig_engine.Engine, Tuple[nn.Module, ...], Dict[str, optim.Optimizer]]:
    """
    Create trainer.

    :param name: Name of this engine.
    :param model: The model to train.
    :param device: The device that the model is using.
    :param _: Configuration.
    :return: `(trainer engine, criteria, optimizers)`
    """
    logger = logging.getLogger(name)
    
    # loss functions
    logger.debug("Using Cross Entropy loss function for classifier.")
    criterion_clas = nn.CrossEntropyLoss()
    
    # optimizer
    logger.debug("Using Adam optimizer with default parameters.")
    optimizer = optim.Adam(
            model.parameters())
    optimizers = {'optimizer': optimizer}
    
    def step(_: ig_engine.Engine, batch):
        """
        The main trainer engine function.
        """
        # unpack batch
        samples, labels, *_ = batch
        samples = samples.to(device)
        labels = labels.to(device)
        
        # set the module in training mode.
        model.train()
        
        # get net outputs
        labels_pred, _ = model(samples)
        
        # calculate loss
        loss = criterion_clas(labels_pred, labels)
        
        # backward propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return dict(loss = loss.item())
    
    # construct engine
    trainer = utils.create_engine_with_logger(step, name)
    
    # running average loss
    ig_metrics.RunningAverage(output_transform = lambda x: x['loss']).attach(trainer, 'loss_classifier')
    
    return trainer, (criterion_clas,), optimizers


def create_validator(
        name: str, model: nn.Module, device: torch.device,
        criteria: Tuple[nn.Module, ...],
        _: configs.Configuration) -> ig_engine.Engine:
    """

    The main difference between validator and tester is that loss are collected in validator.

    :param name: Name of this engine.
    :param model: The model to validate.
    :param device: The device that the model is using.
    :param criteria: The loss functions.
    :param _: Configuration.
    :return: A validator engine.
    """
    _ = logging.getLogger(name)
    
    # loss functions
    criterion_clas, = criteria
    
    def inference(_: ig_engine.Engine, batch):
        """
        The main validator engine function, processing a batch of samples.
        """
        # unpack batch
        samples, labels, *_ = batch
        samples = samples.to(device)
        labels = labels.to(device)
        
        # Sets the module in evaluation mode.
        model.eval()
        
        with torch.no_grad():
            # get net outputs
            labels_pred, labels_pred_soma = model(samples)
        
        return dict(labels = labels, labels_pred = labels_pred, labels_pred_softmaxed = labels_pred_soma)
    
    # construct engine
    validator = utils.create_engine_with_logger(inference, name)
    
    # attach metrics for classification task
    _ = utils.attach_common_metrics(validator, utils.acc_out_transform, device)
    # average loss
    # losses are calculated when evaluating the model for validation purpose
    loss = ig_metrics.Loss(
            criterion_clas,
            output_transform = lambda x: (x['labels_pred'], x['labels']),
            device = device)
    loss.attach(validator, 'loss_classifier')
    
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
            _, labels_pred_soma = model(samples)
        
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
