import argparse
import copy
import os
import pathlib
from typing import NoReturn, Optional, Union

import torch
from ignite import engine as ig_engine, handlers as ig_handlers
from ignite.contrib import handlers as ig_c_handlers

import base_process
import configs
import configs2
import data_loader
import ignite_handlers
from models import base as network_base, mine as network_mine, wicar as network_wicar


class Process(base_process.BaseProcess):
    """
    Train, test.
    """
    
    def __init__(self, conf: configs.Configuration, device: torch.device, network, seed: Optional[int] = None,
                 new_output_folder: bool = True, output_prefix: str = ""):
        """
        
        :param conf: Configurations.
        :param device: The device that model will run on.
        :param network: A module that contains Network, create_trainer, create_validator and create_tester.
        :param seed: The seed.
        :param new_output_folder: Create new output folder under output directory.
        :param output_prefix: Prefix of output folder.
        """
        super(Process, self).__init__(device, conf.output_dir, conf.log_name, conf.log_to_tensorboard, seed,
                                      new_output_folder, output_prefix)
        self.conf = copy.deepcopy(conf)
        self.network = network
        
        # initialize model
        self.model = self.network.Network(self.conf.network).to(self.device)
        
        self.log_metrics_to_file = conf.log_metrics_to_file
        
        # The best validation accuracy that model achieves, and the saved best checkpoint file name.
        self.best_validation_performance: Optional[ignite_handlers.BestPerformance.Item] = None
        self.best_model_path: Optional[pathlib.Path] = None
        self.best_validation_performance_handler: Optional[ignite_handlers.BestPerformance] = None
        self.best_validation_performance_model_saver: Optional[ig_handlers.Checkpoint] = None
    
    def load_data(self, batch_size: int, worker_num: int,
                  target_room: int, additional_labels: bool = False, **kwargs) -> NoReturn:
        """
        Load data to train_loader and test_loader.
        
        :param batch_size: How many samples per batch to load.
        :param worker_num: how many sub-processes to use for data loading. ``0`` means that the data will be loaded in
            the main process.
        :param target_room: The target room.
        :param additional_labels: Should the dataset return additional labels?
        """
        self.logger.info("Batch size is {}.".format(batch_size))
        train_set, test_set = data_loader.get_datasets(
                self.conf.data_dir,
                class_num = self.conf.class_num,
                gram_type = self.conf.gram_type,
                rooms_list = [[i for i in range(self.conf.data_count['room']) if i != target_room], [target_room]],
                additional_labels = additional_labels,
                slicing = self.conf.slicing,
                slice_length = self.conf.slice_length, slice_stride = self.conf.slice_stride)
        self.train_loader = data_loader.get_data_loader(train_set, batch_size = batch_size,
                                                        worker_num = worker_num)
        self.test_loader = data_loader.get_data_loader(test_set, batch_size = batch_size,
                                                       worker_num = worker_num)

        # self.logger.info("Batch size is {}.".format(batch_size))
        # data_set, *_ = data_loader.get_datasets(
        #         self.conf.data_dir,
        #         class_num = self.conf.class_num,
        #         gram_type = self.conf.gram_type,
        #         rooms_list = [[i for i in range(4)]],
        #         additional_labels = additional_labels,
        #         slicing = self.conf.slicing,
        #         slice_length = self.conf.slice_length, slice_stride = self.conf.slice_stride)
        # val_data_set, train_data_set = data_loader.split_dataset(data_set, 0.2)
        # self.train_loader = data_loader.get_data_loader(train_data_set, batch_size, False, 1)
        # self.test_loader = data_loader.get_data_loader(val_data_set, batch_size, False, 1)
    
    def create_trainer(self) -> dict:
        """
        Create trainer, train evaluator and validator.
        
        :return: optimizers.
        """
        self.engine_names['trainer'] = dict(name = "trainer", action_name = "train")
        self.engine_names['train_evaluator'] = dict(name = "train_evaluator", action_name = "train evaluation")
        self.engine_names['validator'] = dict(name = "validator", action_name = "validation")
        self.trainer, criteria, optimizers = self.network.create_trainer(
                self.engine_names['trainer']['name'],
                self.model, self.device,
                self.conf)
        self.train_evaluator = self.network.create_validator(
                self.engine_names['train_evaluator']['name'],
                self.model, self.device,
                criteria, self.conf)
        self.validator = self.network.create_validator(
                self.engine_names['validator']['name'],
                self.model, self.device,
                criteria, self.conf)
        return optimizers
    
    def create_tester(self) -> NoReturn:
        """
        Create tester.
        """
        self.engine_names['tester'] = dict(name = "tester", action_name = "test")
        self.tester = self.network.create_tester(
                self.engine_names['tester']['name'], self.model, self.device,
                self.conf.class_num)
    
    def setup_trainer(self) -> NoReturn:
        """
        Setup trainer, train evaluator and validator.
        """
        optimizers = self.create_trainer()
        
        # time engines
        ignite_handlers.attach_timer(self.trainer, self.engine_names['trainer']['name'],
                                     self.progress_bar)
        ignite_handlers.attach_timer(self.train_evaluator, self.engine_names['train_evaluator']['name'],
                                     self.progress_bar)
        ignite_handlers.attach_timer(self.validator, self.engine_names['validator']['name'],
                                     self.progress_bar)
        
        # attach to progress bar
        if self.progress_bar:
            self.progress_bar.attach(self.trainer, metric_names = "all")
            self.progress_bar.attach(self.train_evaluator)
            self.progress_bar.attach(self.validator)
        
        # save model with the best validation accuracy
        if self.conf.save_models:
            self.best_validation_performance_model_saver = ig_handlers.Checkpoint(
                    to_save = {'model': self.model},
                    save_handler = ig_handlers.DiskSaver(str(self.output_dir), require_empty = False),
                    filename_prefix = 'best',
                    score_function = lambda engine: engine.state.metrics['accuracy'],
                    score_name = "val_acc",
                    global_step_transform = ig_handlers.global_step_from_engine(self.trainer),
                    n_saved = self.conf.checkpoint_max_kept)
            self.validator.add_event_handler(ig_engine.Events.COMPLETED, self.best_validation_performance_model_saver)
        
        # monitor the best validation accuracy
        self.best_validation_performance_handler = ignite_handlers.BestPerformance(
                score_function = lambda engine: engine.state.metrics['accuracy'],
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer),
                num_kept = self.conf.checkpoint_max_kept)
        self.validator.add_event_handler(ig_engine.Events.COMPLETED, self.best_validation_performance_handler)
        
        if self.progress_bar or self.log_metrics_to_file:
            # log metrics
            log_train_metrics = ignite_handlers.MetricsLogger(
                    metric_names = "all",
                    global_step_transform = ig_handlers.global_step_from_engine(self.trainer),
                    progress_bar = self.progress_bar,
                    message_header = self.engine_names['train_evaluator']['action_name'],
                    file_path = self.output_dir / self.conf.train_metrics_name if self.log_metrics_to_file else None)
            self.train_evaluator.add_event_handler(ig_engine.Events.EPOCH_COMPLETED, log_train_metrics)
            log_validator_metrics = ignite_handlers.MetricsLogger(
                    metric_names = "all",
                    global_step_transform = ig_handlers.global_step_from_engine(self.trainer),
                    progress_bar = self.progress_bar,
                    message_header = self.engine_names['validator']['action_name'],
                    file_path = self.output_dir / self.conf.validation_metrics_name if self.log_metrics_to_file else
                    None)
            self.validator.add_event_handler(ig_engine.Events.EPOCH_COMPLETED, log_validator_metrics)
        if self.tb_logger:
            # log metrics to tensorboard
            self.tb_logger.attach(
                    self.trainer,
                    log_handler = ig_c_handlers.tensorboard_logger.OutputHandler(
                            tag = self.engine_names['trainer']['action_name'].replace(' ', '_'),
                            metric_names = "all",
                            global_step_transform = ig_handlers.global_step_from_engine(self.trainer)),
                    event_name = ig_engine.Events.ITERATION_COMPLETED)
            self.tb_logger.attach(
                    self.train_evaluator,
                    log_handler = ig_c_handlers.tensorboard_logger.OutputHandler(
                            tag = self.engine_names['train_evaluator']['action_name'].replace(' ', '_'),
                            metric_names = "all",
                            global_step_transform = ig_handlers.global_step_from_engine(self.trainer)),
                    event_name = ig_engine.Events.EPOCH_COMPLETED)
            self.tb_logger.attach(
                    self.validator,
                    log_handler = ig_c_handlers.tensorboard_logger.OutputHandler(
                            tag = self.engine_names['validator']['action_name'].replace(' ', '_'),
                            metric_names = "all",
                            global_step_transform = ig_handlers.global_step_from_engine(self.trainer)),
                    event_name = ig_engine.Events.EPOCH_COMPLETED)
            # log model parameters to tensorboard
            self.tb_logger.attach(
                    self.trainer,
                    log_handler = ig_c_handlers.tensorboard_logger.WeightsScalarHandler(self.model),
                    event_name = ig_engine.Events.ITERATION_COMPLETED(every = 10))
            self.tb_logger.attach(
                    self.trainer,
                    log_handler = ig_c_handlers.tensorboard_logger.WeightsHistHandler(self.model),
                    event_name = ig_engine.Events.EPOCH_COMPLETED)
            self.tb_logger.attach(
                    self.trainer,
                    log_handler = ig_c_handlers.tensorboard_logger.GradsScalarHandler(self.model),
                    event_name = ig_engine.Events.ITERATION_COMPLETED(every = 10))
            self.tb_logger.attach(
                    self.trainer,
                    log_handler = ig_c_handlers.tensorboard_logger.GradsHistHandler(self.model),
                    event_name = ig_engine.Events.EPOCH_COMPLETED)
            # log optimizer learning rate to tensorboard
            # fixme useless since Adam real lr are calculate within step and all params are fixed.
            # for name, optimizer in optimizers.items():
            #     tb_logger.attach(
            #             trainer,
            #             log_handler = ig_c_handlers.tensorboard_logger.OptimizerParamsHandler(
            #                     optimizer,
            #                     tag = name),
            #             event_name = ig_engine.Events.ITERATION_COMPLETED(every = 10))
        
        # checkpoint objects
        self.checkpoint_objects = {'trainer': self.trainer, 'model': self.model}
        for key, value in optimizers.items():
            self.checkpoint_objects.update({'optimizer_{0}'.format(key): value})
        # save model every checkpoint_interval epoch
        if self.conf.save_models:
            checkpoint1_handler = ig_handlers.Checkpoint(
                    to_save = self.checkpoint_objects,
                    save_handler = ig_handlers.DiskSaver(str(self.output_dir), require_empty = False),
                    filename_prefix = self.conf.checkpoint_prefix,
                    n_saved = self.conf.checkpoint_max_kept)
            self.trainer.add_event_handler(ig_engine.Events.EPOCH_COMPLETED(every = self.conf.checkpoint_interval),
                                           checkpoint1_handler)
        
        # validate every epoch
        @self.trainer.on(ig_engine.Events.EPOCH_COMPLETED)
        def run_evaluation(engine: ig_engine.Engine):
            """
            Run train evaluator and validation.

            :param engine: The trainer.
            """
            message = "{} for epoch {} starts.".format(self.engine_names['train_evaluator']['action_name'].capitalize(),
                                                       engine.state.epoch)
            self.logger.info(message)
            # if self.progress_bar:
            #     self.progress_bar.log_message(message)
            self.train_evaluator.run(self.train_loader)
            
            message = "{} for epoch {} starts.".format(self.engine_names['validator']['action_name'].capitalize(),
                                                       engine.state.epoch)
            self.logger.info(message)
            # if self.progress_bar:
            #     self.progress_bar.log_message(message)
            self.validator.run(self.test_loader)
        
        super(Process, self).setup_trainer()
    
    def setup_tester(self) -> NoReturn:
        """
        Setup tester.
        """
        self.create_tester()
        
        # time engines
        ignite_handlers.attach_timer(self.tester, self.engine_names['tester']['name'],
                                     self.progress_bar)
        
        # attach to progress bar
        
        if self.progress_bar:
            self.progress_bar.attach(self.tester)
        
        if self.progress_bar or self.log_metrics_to_file:
            # log metrics
            log_metrics = ignite_handlers.MetricsLogger(
                    metric_names = "all",
                    progress_bar = self.progress_bar,
                    message_header = self.engine_names['tester']['action_name'],
                    file_path = self.output_dir / self.conf.test_metrics_name if self.log_metrics_to_file else None)
            self.tester.add_event_handler(ig_engine.Events.EPOCH_COMPLETED, log_metrics)
        
        super(Process, self).setup_tester()
    
    def train(self,
              max_epochs: int,
              resume_from: Union[str, pathlib.Path, None] = None,
              saved_model: Union[str, pathlib.Path, None] = None,
              **kwargs) -> NoReturn:
        """
        Train and validation.
        
        :param max_epochs:
        :param resume_from: If not None, the saved model checkpoint which the training will be resumed from.
        :param saved_model: The path to pre-trained model file.
        """
        super(Process, self).train(max_epochs, resume_from, saved_model, **kwargs)
        
        self.best_validation_performance = self.best_validation_performance_handler.kept[0]
        message = "Best validation accuracy {} achieved at epoch {}.".format(self.best_validation_performance.score,
                                                                             self.best_validation_performance.epoch)
        self.logger.info(message)
        if self.progress_bar:
            self.progress_bar.log_message(message)
        if self.conf.save_models:
            self.best_model_path = self.output_dir / self.best_validation_performance_model_saver.last_checkpoint


class FineTuneProcess(Process):
    """
    Fine-tune.  # todo random activity
    """
    
    def load_data(self, batch_size: int, worker_num: int,
                  target_room: int, additional_labels: bool = False, train_test_ratio: float = 0.2) -> NoReturn:
        """
        Load data to train_loader and test_loader.
        
        :param batch_size: How many samples per batch to load.
        :param worker_num: how many sub-processes to use for data loading. ``0`` means that the data will be loaded in
            the main process.
        :param target_room: The target room.
        :param additional_labels: Should the dataset return additional labels?
        :param train_test_ratio: Train/test dataset split ratio.
        """
        self.logger.info("Batch size is {}.".format(batch_size))
        self.logger.info("Train / test dataset split ratio is {}.".format(train_test_ratio))
        dataset, = data_loader.get_datasets(
                self.conf.data_dir,
                class_num = self.conf.class_num,
                gram_type = self.conf.gram_type,
                rooms_list = [[target_room]],
                additional_labels = additional_labels,
                slicing = self.conf.slicing,
                slice_length = self.conf.slice_length, slice_stride = self.conf.slice_stride)
        train_set, test_set = data_loader.split_dataset(dataset, train_test_ratio)
        self.train_loader = data_loader.get_data_loader(train_set, batch_size = self.conf.batch_size,
                                                        worker_num = worker_num)
        self.test_loader = data_loader.get_data_loader(test_set, batch_size = self.conf.batch_size,
                                                       worker_num = worker_num)
    
    def create_trainer(self) -> tuple:
        """
        Create trainer, train evaluator and validator.
        
        :return: optimizers, trainer_name, train_evaluator_name, validator_name.
        """
        self.engine_names['trainer'] = dict(name = "trainer", action_name = "train")
        self.engine_names['train_evaluator'] = dict(name = "train_evaluator", action_name = "train evaluation")
        self.engine_names['validator'] = dict(name = "validator", action_name = "validation")
        self.trainer, criteria, optimizers = self.network.create_fine_tune_trainer(
                self.engine_names['trainer']['name'],
                self.model, self.device,
                self.conf)
        self.train_evaluator = self.network.create_fine_tune_validator(
                self.engine_names['train_evaluator']['name'],
                self.model, self.device,
                criteria, self.conf)
        self.validator = self.network.create_validator(
                self.engine_names['validator']['name'],
                self.model, self.device,
                criteria, self.conf)
        return optimizers


def main(
        data_dir: pathlib.Path,
        output_dir: pathlib.Path,
        batch_size: int,
        epochs: int,
        target_room: int,
        model_name: str = 'my',
        action: str = 'full',
        worker_num: int = 1,
        fine_tune: bool = False,
        train_test_ratio: float = 0.2,
        resume_from: Optional[str] = None,
        saved_model: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None) -> NoReturn:
    """
    Main.
    
    :param data_dir:
    :param output_dir:
    :param batch_size:
    :param epochs:
    :param target_room:
    :param model_name:
    :param action:
    :param worker_num:
    :param fine_tune:
    :param train_test_ratio:
    :param resume_from:
    :param saved_model:
    :param seed:
    :param device:
    """
    if model_name == 'my':
        network = network_mine
    elif model_name == 'wicar':
        if fine_tune:
            raise NotImplementedError("Model {} can not be fine-tuned.".format(model_name))
        network = network_wicar
    elif model_name == 'base':
        if fine_tune:
            raise NotImplementedError("Model {} can not be fine-tuned.".format(model_name))
        network = network_base
    else:
        raise ValueError("Unknown model {}".format(model_name))
    
    network_configuration = network.NetworkConfiguration()
    conf = configs.Configuration(
            data_dir = data_dir, output_dir = output_dir,
            train_test_ratio = train_test_ratio,
            batch_size = batch_size, max_epochs = epochs,
            network = network_configuration)
    if model_name in ["wicar"]:
        conf.slicing = False
    if model_name in ["wicar"]:
        additional_labels = True
    else:
        additional_labels = False
    
    if fine_tune:
        process = FineTuneProcess(conf, device, network, seed,
                                  f'{network.__name__}_room{target_room}_split{train_test_ratio}')
        process.load_data(batch_size, worker_num = worker_num,
                          target_room = target_room,
                          additional_labels = additional_labels,
                          train_test_ratio = train_test_ratio)
    else:
        process = Process(conf, device, network, seed,
                          f'{network.__name__}_room{target_room}')
        process.load_data(batch_size, worker_num = worker_num,
                          target_room = target_room,
                          additional_labels = additional_labels)
    
    process.print_model()
    
    if fine_tune or action == 'test':
        process.load_model(saved_model)
    if action in ['full', 'train']:
        process.setup_trainer()
        process.train(epochs, resume_from)
    if action in ['full', 'test']:
        process.setup_tester()
        process.test()
    process.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-dir", default = str(pathlib.Path('/workspace/data/gram_data_widar/all_gram')),
                        help = "path to dataset.")
    parser.add_argument("--output-dir", default = str(configs.default_configs.output_dir),
                        help = "path to output logs.")
    parser.add_argument("--batch-size", type = int, default = configs.default_configs.batch_size,
                        help = "input batch size.")
    parser.add_argument("--epochs", type = int, default = configs.default_configs.max_epochs,
                        help = "number of epochs to train for.")
    parser.add_argument("--model", type = str, default = 'my',
                        help = "The model to use, 'my', 'base', 'wicar'.")
    parser.add_argument("--train", action = "store_true",
                        help = "only train the model. default action is train and test.")
    parser.add_argument("--test", action = "store_true",
                        help = "only test the model. default action is train and test")
    parser.add_argument("--workers", type = int, default = 1,
                        help = "number of data loader workers. If 0, then data loader will use main thread to load "
                               "data.")
    parser.add_argument("--target-room", type = int,
                        help = "The target room domain, 0, 1, 2, 3.")
    parser.add_argument("--fine-tune", action = "store_true",
                        help = "fine-tune the saved model (--saved-model) on target room.")
    parser.add_argument("--train-test-ratio", type = float, default = configs.default_configs.train_test_ratio,
                        help = "the train test data split ratio when fine-tuning.")
    parser.add_argument("--saved-model", default = "",
                        help = "the saved model to start fine-tune or test from.")
    parser.add_argument("--resume-from", default = "",
                        help = "path to pickled network (to continue training).")
    parser.add_argument("--seed", type = int,
                        help = "manual seed.")
    parser.add_argument("--no-cuda", action = "store_true",
                        help = "disables cuda.")
    
    arguments = parser.parse_args()
    
    dev = torch.device("cpu" if (not torch.cuda.is_available() or arguments.no_cuda) else "cuda:1")

    try:
        os.makedirs(arguments.output_dir)
    except FileExistsError:
        if not os.path.isdir(arguments.output_dir):
            raise FileExistsError("Please provide a path to a non-existing or directory as the output directory.")
    
    if arguments.train and not arguments.test:
        act = 'train'
    elif not arguments.train and arguments.test:
        act = 'test'
    else:
        act = 'full'
    
    if arguments.fine_tune:
        if arguments.resume_from == "" and arguments.saved_model == "":
            raise ValueError("Please provide saved-model or resume-from to start fine-tune.")
    if arguments.target_room:
        target_rooms = [arguments.target_room]
    else:
        target_rooms = [room for room in range(configs2.data_count['room'])]
    
    for room in target_rooms:
        if arguments.fine_tune:
            main(data_dir = pathlib.Path(arguments.data_dir),
                 output_dir = pathlib.Path(arguments.output_dir),
                 batch_size = arguments.batch_size,
                 epochs = arguments.epochs,
                 model_name = arguments.model,
                 action = act,
                 worker_num = arguments.workers,
                 target_room = room,
                 fine_tune = True,
                 train_test_ratio = arguments.train_test_ratio,
                 resume_from = arguments.resume_from,
                 saved_model = arguments.saved_model,
                 seed = arguments.seed,
                 device = dev)
        else:
            main(data_dir = pathlib.Path(arguments.data_dir),
                 output_dir = pathlib.Path(arguments.output_dir),
                 batch_size = arguments.batch_size,
                 epochs = arguments.epochs,
                 model_name = arguments.model,
                 action = act,
                 worker_num = arguments.workers,
                 target_room = room,
                 resume_from = arguments.resume_from,
                 seed = arguments.seed,
                 device = dev)
