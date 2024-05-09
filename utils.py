"""
Utility functions and classes.
"""
import csv
import logging
import logging.config
import pathlib
import random
from typing import Any, Callable, Iterable, NoReturn, Optional

import numpy
import pytorch_model_summary
import torch
from ignite import engine as ig_engine, metrics as ig_metrics
from ignite.contrib import handlers as ig_c_handlers
from torch import nn as nn


def manual_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """
    If seed is not given manually, generate a random one and log it.
    
    :param seed: The seed.
    :param deterministic: Set cuDNN to deterministic implementation.
    :return: The seed.
    """
    seed = seed or random.randint(0, 1000000000)
    
    random.seed(seed)
    seed1 = random.randint(0, 1000000000)
    torch.manual_seed(seed1)
    seed2 = random.randint(0, 1000000000)
    numpy.random.seed(seed2)
    
    logging.getLogger('main').info(
            "Use seed {seed}, pytorch seed {seed1}, numpy seed {seed2}.".format(seed = seed, seed1 = seed1,
                                                                                seed2 = seed2))
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.getLogger('main').info("Use deterministic cuDNN Algorithm.")
    else:
        # faster convolution
        torch.backends.cudnn.benchmark = True
        logging.getLogger('main').info("Use faster cuDNN Algorithm.")
    return seed


def acc_out_transform(output: dict):
    """
    Transform the :class:`ignite.engine.Engine`'s `process_function`'s output into the form expected by metrics of
    classifier.
    
    Should use softmaxed prediction to compute accuracy, precision, et c.
    
    :param output: The output of :class:`ignite.engine.Engine`'s `process_function`.
    :return: y_pred, y
    """
    y_pred = output['labels_pred_softmaxed']
    y = output['labels']
    return y_pred, y


def domain_acc_out_transform(output: dict):
    """
    Transform the :class:`ignite.engine.Engine`'s `process_function`'s output into the form expected by metrics of
    domain discriminator.
    
    Should use softmaxed prediction to compute accuracy, precision, et c.
    
    :param output: The output of :class:`ignite.engine.Engine`'s `process_function`.
    :return: y_pred, y
    """
    y_pred = output['domain_labels_pred_softmaxed']
    y = output['domain_labels']
    return y_pred, y


def create_engine_with_logger(process_function: Callable[[ig_engine.Engine], Any], name: str) -> ig_engine.Engine:
    """
    Create an :class:`ignite.engine.Engine` and configure its logger.

    :param process_function: Process function
    :param name: Engine logger's name will be name.engine.
    :return: the Engine
    """
    engine = ig_engine.Engine(process_function)
    engine.logger = logging.getLogger(name + '.engine')
    engine.logger.setLevel(logging.INFO)
    return engine


def attach_common_metrics(engine: ig_engine.Engine, out_transform, device: torch.device, name_prefix: str = '') -> list:
    """
    Attach evaluation metrics to the engine.
    
    :param engine: The engine to attach metrics to.
    :param out_transform: The output_transform function passed to ignite metrics.
    :param name_prefix: The prefix of metrics names.
    :param device: The device that the engine is running on.
    :return: A list of metrics names.
    """
    name_prefix = name_prefix + '_' if len(name_prefix) > 0 else name_prefix
    acc = ig_metrics.Accuracy(output_transform = out_transform, device = device)
    acc.attach(engine, name_prefix + 'accuracy')
    prec = ig_metrics.Precision(output_transform = out_transform, average = True, device = device)
    prec.attach(engine, name_prefix + 'precision')
    recall = ig_metrics.Recall(output_transform = out_transform, average = True, device = device)
    recall.attach(engine, name_prefix + 'recall')
    f1 = ig_metrics.Fbeta(1, output_transform = out_transform, device = device)
    f1.attach(engine, name_prefix + 'f1')
    return [name_prefix + 'accuracy', name_prefix + 'precision', name_prefix + 'recall', name_prefix + 'f1']


def print_model(model: nn.Module, sample: torch.Tensor,
                logger: logging.Logger, to_console: bool = True,
                tb_logger: Optional[ig_c_handlers.TensorboardLogger] = None) -> NoReturn:
    """
    Print and log the model.
    
    :param model: The model.
    :param sample: The sample to feed into model.
    :param logger: Logger.
    :param to_console: Print model to console.
    :param tb_logger: Tensorboard logger.
    """
    summary = pytorch_model_summary.summary(model, sample, show_input = True, max_depth = 8)
    if to_console:
        print(summary)
    logger.debug(summary)
    summary = pytorch_model_summary.summary(model, sample, show_hierarchical = True, max_depth = 8)
    if to_console:
        print(summary)
    logger.debug(summary)
    if to_console:
        print(model)
    logger.debug(repr(model))
    if tb_logger:
        # print model to tensorboard
        tb_logger.writer.add_graph(model, sample)


class CsvSaver:
    """
    Save data to csv file.
    """
    
    def __init__(self, file_path: pathlib.Path, append = False) -> NoReturn:
        """
        
        :param file_path: Path to the csv file.
        """
        if append:
            self.csv = file_path.open(mode = 'r', newline = '')
            reader = csv.DictReader(self.csv, dialect = 'excel-tab')
            column_names = reader.fieldnames
            self.csv.close()
            del reader
            self.csv = file_path.open(mode = 'a', newline = '', buffering = 1)
            self.csv_writer = csv.DictWriter(self.csv, column_names, dialect = 'excel-tab')
        else:
            self.csv = file_path.open(mode = 'w', newline = '', buffering = 1)
            self.csv_writer = None
        self.data = []
    
    def write_row(self, datum_dict: dict) -> NoReturn:
        """
        Write an entry of data.
        
        :param datum_dict: Datum dictionary.
        """
        if not self.csv_writer:
            self.write_header(datum_dict.keys())
        
        self.data.append(datum_dict)
        self.csv_writer.writerow(datum_dict)
        self.csv.flush()
    
    def write_header(self, column_names: Iterable) -> NoReturn:
        """
        Write column headers.
        
        :param column_names: Column names.
        """
        self.csv_writer = csv.DictWriter(self.csv, column_names, dialect = 'excel-tab')
        self.csv_writer.writeheader()
        self.csv.flush()
    
    def open(self, file_path: pathlib.Path) -> NoReturn:
        """
        Open file.
        
        :param file_path: Path to the csv file.
        """
        if self.csv and not self.csv.closed:
            self.close()
        
        self.csv = file_path.open(mode = 'w', newline = '')
        self.csv_writer = None
    
    def close(self) -> NoReturn:
        """
        Close file.
        """
        self.csv.close()
        self.csv_writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
