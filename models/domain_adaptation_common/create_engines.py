import logging

import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from torch import nn as nn

import utils


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
            _, labels_pred_soma = model(samples, mode = 'base')
        
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
