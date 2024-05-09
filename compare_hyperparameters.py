import os
import pathlib

import torch
from ignite import engine as ig_engine, handlers as ig_handlers
from ray import tune as tune
from ray.tune import suggest as suggests

import ignite_handlers
from configs import (Configuration as Configuration, Loss as LossConfiguration,
                     Optimizer as OptimizerConfiguration)
from models import mine as network
from train import Process


log_format = '%(asctime)s: %(levelname)s: %(name)s: %(filename)s: %(funcName)s(): %(lineno)d:\t%(message)s'
seeds = []
worker_num = 2


class Trainable(tune.Trainable):
    """
    Trainable

    """
    
    def _setup(self, config: dict):
        """

        :param config: A dict of hyper-parameters.
        """
        data_dir = config['data_dir']
        output_dir = pathlib.Path('.')
        batch_size = config['batch_size']
        self.epoch = config['epoch']
        
        loss_conf = LossConfiguration()
        if 'label_smoothening' in config:
            loss_conf.classifier_label_smoothen = (config['label_smoothening'] != 0.0)
            loss_conf.classifier_label_smooth_factor = config['label_smoothening']
            loss_conf.discriminator_label_smoothen = (config['label_smoothening'] != 0.0)
            loss_conf.discriminator_label_smooth_factor = config['label_smoothening']
        
        if 'lambda_smoothness' in config:
            loss_conf.regularization_smooth = (config['lambda_smoothness'] != 0.0)
            loss_conf.lambda_smooth = config['lambda_smoothness']
        
        optim_conf = OptimizerConfiguration()
        if 'learning_rate' in config:
            optim_conf.learning_rate = config['learning_rate']
        if 'learning_rate_disc' in config:
            optim_conf.learning_rate_discriminator = config['learning_rate_disc']
        
        net_conf = network.NetworkConfiguration()
        
        self.conf = Configuration(
                data_dir = data_dir, output_dir = output_dir, batch_size = batch_size, max_epochs = self.epoch,
                loss = loss_conf, optimizer = optim_conf, network = net_conf,
                save_models = False, log_metrics_to_file = True, log_to_tensorboard = True)
        
        if 'gram_type' in config:
            self.conf.gram_type = config['gram_type']
        if self.conf.gram_type == 'ampl':  # amplitude gram only have 3 channel
            self.conf.gram_channel_num = self.conf.network.local_extractor['in_channel_num'] = 3
        
        if 'slice_length' in config:
            self.conf.slice_length = config['slice_length']
            self.conf.slice_stride = self.conf.slice_length // 2
        
        self.device = torch.device('cuda')
        self.seed = config['seed']  # .get('seed', seeds[config.get(suggests.repeater.TRIAL_INDEX, 0)])
        self.test_room = config['test_room']
        
        self.process = Process(self.conf, self.device, network, self.seed, new_output_folder = False)
        self.process.set_seed(self.seed, deterministic = False)
        self.process.progress_bar = None
        self.process.load_data(batch_size, worker_num = worker_num, target_room = self.test_room,
                               additional_labels = config.get('additional_labels', False))
        self.process.print_model(to_console = False)
        self.process.setup_trainer()
        self.process.setup_tester()
        
        # Record metric history
        self.train_recorder = ignite_handlers.HistoryRecorder(
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.process.trainer))
        self.process.train_evaluator.add_event_handler(ig_engine.Events.COMPLETED, self.train_recorder)
        self.recorder = ignite_handlers.HistoryRecorder(
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.process.trainer))
        self.process.validator.add_event_handler(ig_engine.Events.COMPLETED, self.recorder)
        
        self.iteration_count = 0
    
    def _train(self) -> dict:
        self.iteration_count += 1
        self.process.train(max_epochs = self.iteration_count * self.epoch)
        
        mean_train_metrics = self.train_recorder.get(self.epoch * (self.iteration_count - 1) + 1,
                                                     self.epoch * self.iteration_count + 1, 'mean')
        mean_metrics = self.recorder.get(self.epoch * (self.iteration_count - 1) + 1,
                                         self.epoch * self.iteration_count + 1, 'mean')
        
        return {"mean_accuracy": mean_metrics['accuracy'],
                "mean_loss": mean_metrics['loss_total'],
                "mean_metrics": mean_metrics,
                "train_mean_metrics": mean_train_metrics,
                "best_accuracy": self.process.best_validation_performance.score,
                "best_performance_epoch": self.process.best_validation_performance.epoch,
                "best_metrics": self.process.best_validation_performance.metrics}
    
    def _save(self, checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
        self.process.save_model(save_path = checkpoint_path)
        return checkpoint_path
    
    def _restore(self, checkpoint: str):
        self.process.load_model(checkpoint)
