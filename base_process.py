import abc
import importlib
import logging
import logging.config
import os
import pathlib
from datetime import datetime
from typing import Any, Dict, NoReturn, Optional, Union

import torch
import yaml
from ignite import engine as ig_engine, handlers as ig_handlers
from ignite.contrib import handlers as ig_c_handlers
from torch import nn
from torch.utils import data as torch_data

import utils


class BaseProcess(abc.ABC):
    """
    Base process.
    """
    
    def __init__(self, device: torch.device,
                 output_dir: pathlib.Path,
                 log_name: str,
                 log_to_tensorboard: bool = False,
                 seed: Optional[int] = None,
                 new_output_folder: bool = True,
                 output_prefix: str = "") -> NoReturn:
        """
        
        :param device: The device that model will run on.
        :param output_dir: Output directory.
        :param log_name: Log file name.
        :param log_to_tensorboard: Log to tensorboard or not.
        :param seed: The seed.
        :param new_output_folder: Create new output folder under output directory.
        :param output_prefix: Prefix of output folder.
        """
        # setup loggers
        self.output_dir = self.create_output_dir(output_dir, output_prefix) if new_output_folder else output_dir
        self.config_logging(log_name)
        self.logger = logging.getLogger('main')
        self.tb_logger = ig_c_handlers.TensorboardLogger(log_dir = str(self.output_dir)) if log_to_tensorboard else None
        self.progress_bar = ig_c_handlers.ProgressBar(persist = False)
        self.log_metrics_to_file: Optional[bool] = None
        
        # set device
        self.device = device or torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
        
        # set seed
        self.seed = seed
        self.set_seed(seed)
        
        # model
        self.model: Optional[nn.Module] = None
        
        # data loaders
        self.train_loader: Optional[torch_data.DataLoader] = None
        self.validation_loader: Optional[torch_data.DataLoader] = None
        self.test_loader: Optional[torch_data.DataLoader] = None
        
        # engines
        self.trainer: Optional[ig_engine.Engine] = None
        self.train_evaluator: Optional[ig_engine.Engine] = None
        self.validator: Optional[ig_engine.Engine] = None
        self.tester: Optional[ig_engine.Engine] = None
        self.engine_names: Dict[str, Dict[str, str]] = {}
        self.trainer_ready: bool = False
        self.tester_ready: bool = False
        
        # model checkpoint dictionary
        self.checkpoint_objects: Optional[dict] = None
    
    def print_model(self, to_console: bool = True) -> NoReturn:
        """
        Print and log the model.
        
        :param to_console: Print model to console.
        """
        tmp_iterator = iter(self.test_loader)
        sample, *_ = next(tmp_iterator)
        sample = sample.to(self.device)
        utils.print_model(self.model, sample, self.logger, to_console, self.tb_logger)
        del tmp_iterator, sample
    
    def load_model(self, saved_model_path: Union[str, pathlib.Path]):
        """
        Load pre-trained model
        
        :param saved_model_path: The path to saved model file.
        """
        message = "Load model from {}.".format(saved_model_path)
        self.logger.debug(message)
        if self.progress_bar:
            self.progress_bar.log_message(message)
        checkpoint = torch.load(saved_model_path)
        ig_handlers.Checkpoint.load_objects(to_load = {'model': self.model}, checkpoint = checkpoint)
    
    def save_model(self, save_name: str = None, save_path: Union[str, pathlib.Path] = None):
        """
        Save trained model
        
        :param save_name: The name to save model file under output directory. Only use one of `save_name` and
        `save_path`.
        :param save_path: The path to saved= model file. Only use one of `save_name` and `save_path`.
        """
        if save_name:
            if save_path:
                raise ValueError("Only use one of save_name and save_path.")
            save_path = self.output_dir / save_name
        elif not save_path:
            raise ValueError("Must specify save_name or save_path.")
        
        message = "Save model to {}.".format(save_path)
        self.logger.debug(message)
        if self.progress_bar:
            self.progress_bar.log_message(message)
        torch.save({'model': self.model.state_dict()}, save_path)
    
    def save_checkpoint(self, checkpoint_path: Union[str, pathlib.Path]):  # todo
        """
        Save model training checkpoint.
        
        :param checkpoint_path: The path to save training checkpoint file.
        """
        raise NotImplementedError
    
    def load_checkpoint(self, checkpoint_path: Union[str, pathlib.Path]):
        """
        Load model training checkpoint.
        
        :param checkpoint_path: The path to saved training checkpoint file.
        """
        message = "Resume training from checkpoint {}.".format(checkpoint_path)
        self.logger.debug(message)
        if self.progress_bar:
            self.progress_bar.log_message(message)
        checkpoint = torch.load(checkpoint_path)
        ig_handlers.Checkpoint.load_objects(to_load = self.checkpoint_objects, checkpoint = checkpoint)
    
    @abc.abstractmethod
    def load_data(self, batch_size: int, worker_num: int,
                  *args, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Load data to train_loader, test_loader, et. cetera.
        
        :param batch_size: How many samples per batch to load.
        :param worker_num: how many sub-processes to use for data loading. ``0`` means that the data will be loaded in
            the main process.
        :param args:
        :param kwargs:
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def create_trainer(self, *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Create trainer.
        
        :param args:
        :param kwargs:
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def create_tester(self, *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Create tester.
        
        :param args:
        :param kwargs:
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def setup_trainer(self, *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Setup trainer.
        
        :param args:
        :param kwargs:
        """
        self.trainer_ready = True
    
    @abc.abstractmethod
    def setup_tester(self, *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Setup tester.
        
        :param args:
        :param kwargs:
        """
        self.tester_ready = True
    
    def train(self,
              max_epochs: int,
              resume_from: Union[str, pathlib.Path, None] = None,
              saved_model: Union[str, pathlib.Path, None] = None,
              *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Train.
        
        :param max_epochs: Max training epochs.
        :param resume_from: If not None, the path to saved training checkpoint file where the training will be
        resumed from.
        :param saved_model: The path to pre-trained model file.
        :param args:
        :param kwargs: Keyword arguments passed to :meth:`setup_trainer`.
        """
        if not self.trainer_ready:
            self.setup_trainer(**kwargs)
        
        if resume_from:
            self.load_checkpoint(resume_from)
        elif saved_model:
            self.load_model(saved_model)
        
        # run trainer
        message = "Train starts, run for {} epochs.".format(max_epochs)
        self.logger.info(message)
        if self.progress_bar:
            self.progress_bar.log_message(message)
        self.trainer.run(self.train_loader, max_epochs = max_epochs)
    
    def test(self,
             saved_model: Union[str, pathlib.Path, None] = None,
             *args: tuple, **kwargs: Dict[str, Any]) -> NoReturn:
        """
        Test.
        
        :param saved_model: The path to pre-trained model file.
        :param args:
        :param kwargs: Keyword arguments passed to :meth:`setup_tester`.
        """
        if not self.tester_ready:
            self.setup_tester(**kwargs)
        
        if saved_model:
            self.load_model(saved_model)
        
        # run tester
        self.logger.info("Test starts.")
        if self.progress_bar:
            self.progress_bar.log_message("Test starts.")
        self.tester.run(self.test_loader)
    
    def get_test_result(self) -> dict:
        """
        get test results.
        
        :return: Tester metrics.
        """
        result = dict()
        for key, value in self.tester.state.metrics.items():
            if type(value) == torch.Tensor:
                value = value.tolist()
            else:
                pass
            result[key] = value
        return result
    
    def quit(self) -> NoReturn:
        """
        Work when quit.
        """
        logging.shutdown()
        if self.tb_logger:
            self.tb_logger.close()
    
    def set_seed(self, seed: Optional[int] = None, deterministic: bool = True) -> NoReturn:
        """
        If seed is not given manually, generate a random one and log it.
        
        :param seed: The seed.
        :param deterministic: Set cuDNN to deterministic implementation.
        """
        self.seed = utils.manual_seed(seed, deterministic)
    
    @staticmethod
    def create_output_dir(output_dir: pathlib.Path, output_prefix: str) -> pathlib.Path:
        """
        Create output folder under output directory.
        
        Create a folder named by current time under output_dir.
        
        :param output_dir: Output directory.
        :param output_prefix: Prefix of output folder.
        :return: Logging directory.
        """
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        output_dir /= (output_prefix + current_time)
        os.mkdir(str(output_dir))
        return output_dir
    
    def config_logging(self, log_name: str):
        """
        Configure logging.
        
        :param log_name: Logger logging file name.
        """
        importlib.reload(logging)
        with open('/workspace/project/logging.yml') as log_conf_file:
            log_conf = yaml.load(log_conf_file, Loader = yaml.FullLoader)
        log_conf['handlers']['file']['filename'] = str(self.output_dir / log_name)
        logging.config.dictConfig(log_conf)
