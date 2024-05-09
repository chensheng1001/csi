import argparse
import logging
import logging.config
import os
import pathlib

import ray
import torch
from ax.service import ax_client as ax_client
from ignite import engine as ig_engine, handlers as ig_handlers
from ray import tune as tune
from ray.tune import schedulers as schedulers, suggest as suggests
from ray.tune.suggest import ax as ax_suggest

import data_loader
import ignite_handlers
import utils
from configs import (Configuration as Configuration, Loss as LossConfiguration, Network as NetworkConfiguration,
                     Optimizer as OptimizerConfiguration, hyper_parameter_constraints,
                     hyper_parameters)
from configs2 import activity_class_num, data_count
from models import mine as network

seeds = [9723898637, 2968985764, 8973015667, 9684043765, 442747722]
worker_num = 2


class Trainable(tune.Trainable):
    """
    Trainable
    
    """
    
    def _setup(self, config: dict):
        """

        :param config: A dict of hyper-parameters.
        """
        loss_conf = LossConfiguration()
        if 'classifier_type' in config:
            loss_conf.classifier_type = config['classifier_type']
        if 'classifier_label_smoothing' in config:
            loss_conf.classifier_label_smoothen = (config['classifier_label_smoothing'] != 0.0)
            loss_conf.classifier_label_smooth_factor = config['classifier_label_smoothing']
        if 'discriminator_label_smoothing' in config:
            loss_conf.discriminator_label_smoothen = (config['discriminator_label_smoothing'] != 0.0)
            loss_conf.discriminator_label_smooth_factor = config['discriminator_label_smoothing']
        if 'lambda_disc' in config:
            loss_conf.lambda_disc = config['lambda_disc']
        if 'lambda_smooth' in config:
            loss_conf.regularization_smooth = (config['lambda_smooth'] != 0.0)
            loss_conf.lambda_smooth = config['lambda_smooth']
        if 'smooth_eta' in config:
            loss_conf.smooth_eta = config['smooth_eta']
        if 'smooth_alpha' in config:
            loss_conf.smooth_alpha = config['smooth_alpha']
        
        optim_conf = OptimizerConfiguration()
        if 'learning_rate' in config:
            optim_conf.learning_rate = config['learning_rate']
        if 'beta_1' in config:
            optim_conf.beta_1 = config['beta_1']
        if 'beta_2' in config:
            optim_conf.beta_2 = config['beta_2']
        if 'eps' in config:
            optim_conf.eps = config['eps']
        if 'learning_rate_disc' in config:
            optim_conf.learning_rate_discriminator = config['learning_rate_disc']
        
        net_conf = NetworkConfiguration()
        # if 'local_feature_size' in config:
        #     local_feature_size = config['local_feature_size']
        #     net_conf.local_feature_size = local_feature_size
        # else:
        #     local_feature_size = net_conf.local_feature_size
        # if 'activator_type' in config:
        #     net_conf.activator_type = config['activator_type']
        # if 'local_extractor_norm' in config:
        #     net_conf.local_extractor['norm_position'] = config['local_extractor_norm']
        # if 'local_extractor_layers' in config:
        #     local_extractor_layers = config['local_extractor_layers']
        #     net_conf.local_extractor['layer'] = local_extractor_layers
        # else:
        #     local_extractor_layers = net_conf.local_extractor['layer']
        # if local_feature_size == 64:
        #     if local_extractor_layers == 4:
        #         channel_num = [32 * i for i in [1, 2, 3, 4]]  # [32, 64, 96, 128]
        #     elif local_extractor_layers == 6:
        #         channel_num = [32 * i for i in [1, 2, 3, 3, 4, 4]]  # [32, 64, 96, 96, 128, 128]
        #     # elif local_extractor_layers == 8:
        #     #     channel_num = [32 * i for i in [1, 1, 2, 2, 3, 3, 4, 4]]  # [32, 32, 64, 64, 96, 96, 128, 128]
        #     else:
        #         raise ValueError("Unknown local extractor layers {}.".format(local_extractor_layers))
        # elif local_feature_size == 128:
        #     if local_extractor_layers == 4:
        #         channel_num = [32 * i for i in [1, 2, 4, 6]]  # [32, 64, 128, 192]
        #     elif local_extractor_layers == 6:
        #         channel_num = [32 * i for i in [1, 2, 4, 4, 6, 6]]  # [32, 64, 128, 128, 192, 192]
        #     # elif local_extractor_layers == 8:
        #     #     channel_num = [32 * i for i in [1, 2, 3, 3, 4, 4, 6, 6]]  # [32, 64, 96, 96, 128, 128, 192, 192]
        #     else:
        #         raise ValueError("Unknown local extractor layers {}.".format(local_extractor_layers))
        # elif local_feature_size == 256:
        #     if local_extractor_layers == 4:
        #         channel_num = [32 * i for i in [2, 4, 8, 8]]  # [64, 128, 256, 256]
        #     elif local_extractor_layers == 6:
        #         channel_num = [32 * i for i in [2, 2, 4, 4, 8, 8]]  # [64, 64, 128, 128, 256, 256]
        #     # elif local_extractor_layers == 8:
        #     #     channel_num = [32 * i for i in [2, 2, 4, 6, 8, 8, 9, 9]]  # [64, 128, 192, 192, 256, 256, 288, 288]
        #     else:
        #         raise ValueError("Unknown local extractor layers {}.".format(local_extractor_layers))
        # # elif local_feature_size == 512:
        # #     if local_extractor_layers == 4:
        # #         channel_num = [64 * i for i in [1, 2, 4, 8]]  # [64, 128, 256, 512]
        # #     elif local_extractor_layers == 6:
        # #         channel_num = [64 * i for i in [1, 2, 4, 4, 8, 8]]  # [64, 128, 256, 256, 512, 512]
        # #     elif local_extractor_layers == 8:
        # #         channel_num = [64 * i for i in [1, 1, 2, 2, 4, 4, 8, 8]]  # [64, 64, 128, 128, 256, 256, 512, 512]
        # #     else:
        # #         raise ValueError("Unknown local extractor layers {}.".format(local_extractor_layers))
        # else:
        #     raise ValueError("Unknown local feature size {}.".format(local_feature_size))
        # net_conf.local_extractor['channel_num'] = channel_num
        # if 'local_extractor_pool_kernel_size' in config:
        #     tmp = config['local_extractor_pool_kernel_size']
        #     net_conf.local_extractor['pool_kernel_size'] = [(tmp, tmp)] * local_extractor_layers
        # net_conf.local_extractor['pool_stride'] = [(2, 2)] * local_extractor_layers
        # net_conf.local_extractor['global_pool_channel_num'] = local_feature_size
        # if 'temporal_extractor_type' in config:
        #     net_conf.temporal_extractor['type'] = config['temporal_extractor_type']
        # if 'temporal_extractor_norm' in config:
        #     net_conf.temporal_extractor['norm_position'] = config['temporal_extractor_norm']
        # if 'temporal_extractor_bidirectional' in config:
        #     net_conf.temporal_extractor['bidirectional'] = config['temporal_extractor_bidirectional']
        # if 'temporal_extractor_layers' in config:
        #     net_conf.temporal_extractor['layer'] = config['temporal_extractor_layers']
        # net_conf.temporal_extractor['hidden_size'] = local_feature_size
        # if 'temporal_extractor_dropout' in config:
        #     net_conf.temporal_extractor['dropout'] = config['temporal_extractor_dropout']
        # if net_conf.temporal_extractor['bidirectional']:
        #     net_conf.feature_size = local_feature_size * 2
        # else:
        #     net_conf.feature_size = local_feature_size
        # if 'classifier_norm' in config:
        #     net_conf.classifier['norm_position'] = config['classifier_norm']
        #     net_conf.domain_discriminator['norm_position'] = config['classifier_norm']
        # if 'classifier_layers' in config:
        #     classifier_layers = config['classifier_layers']
        # else:
        #     classifier_layers = net_conf.classifier['layer']
        # net_conf.classifier['layer'] = classifier_layers
        # net_conf.domain_discriminator['layer'] = classifier_layers
        # net_conf.classifier['out_size'] = [local_feature_size] * (classifier_layers - 1) + [
        #         activity_class_num]
        # net_conf.domain_discriminator['out_size'] = [local_feature_size] * (classifier_layers - 1) + [
        #         data_count['room']]
        
        if 'feature_size' in config:
            feature_size = config['feature_size']
            net_conf.feature_size = feature_size
            if net_conf.temporal_extractor['bidirectional']:
                net_conf.local_feature_size = feature_size // 2
            else:
                net_conf.local_feature_size = feature_size
            net_conf.temporal_extractor['hidden_size'] = net_conf.local_feature_size
            tmp = net_conf.local_feature_size // net_conf.local_extractor['global_pool_channel_num']
            if tmp == 1:
                net_conf.local_extractor['global_pool_out_size'] = (1, 1)
            elif tmp == 2:
                net_conf.local_extractor['global_pool_out_size'] = (2, 1)
            elif tmp == 4:
                net_conf.local_extractor['global_pool_out_size'] = (2, 2)
            elif tmp == 8:
                net_conf.local_extractor['global_pool_out_size'] = (4, 2)
            else:
                raise ValueError("local_feature_size / global_pool_channel_num is  {}.".format(tmp))
            net_conf.classifier['out_size'] = [feature_size // 2, feature_size // 4] + [activity_class_num]
            net_conf.domain_discriminator['out_size'] = [feature_size // 2, feature_size // 4] + [data_count['room']]
        if 'classifier_size0' in config:
            net_conf.classifier['out_size'][0] = config['classifier_size0']
            net_conf.domain_discriminator['out_size'][0] = config['classifier_size0']
        if 'classifier_size1' in config:
            net_conf.classifier['out_size'][1] = config['classifier_size1']
            net_conf.domain_discriminator['out_size'][1] = config['classifier_size1']
        if 'domain_feature_type' in config:
            net_conf.domain_feature_type = config['domain_feature_type']
        if net_conf.domain_feature_type == 'concatenation':
            net_conf.domain_feature_size = net_conf.feature_size + net_conf.class_num
        else:
            net_conf.domain_feature_size = net_conf.feature_size
        
        data_dir = config['data_dir']
        output_dir = pathlib.Path('.')
        self.conf = Configuration(
                data_dir = data_dir, output_dir = output_dir,
                max_epochs = config['epoch_per_iteration'], batch_size = config['batch_size'],
                loss = loss_conf, optimizer = optim_conf, network = net_conf,
                save_models = False, log_metrics_to_file = False, log_to_tensorboard = False)
        if 'gram_type' in config:
            self.conf.gram_type = config['gram_type']
        if 'slice_stride' in config:
            self.conf.slice_stride = config['slice_stride']
        if 'slice_length' in config:
            self.conf.slice_length = config['slice_length']
            if 'slice_stride' not in config:
                self.conf.slice_stride = config['slice_length'] // 2
        if 'adversarial_training' in config:
            self.conf.adversarial_training = config['adversarial_training']
        if 'stable_adversarial_training' in config:
            self.conf.stable_adversarial_training = config['stable_adversarial_training']
        
        if self.conf.gram_type == 'ampl':  # amplitude gram only have 3 channel
            self.conf.gram_channel_num = self.conf.network.local_extractor['in_channel_num'] = 3
        
        self.epoch_per_iteration = config['epoch_per_iteration']
        self.device = torch.device('cuda')
        
        # config logging
        trial_id = str(self.trial_id)
        logger = logging.getLogger(trial_id)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        file_handler = logging.FileHandler('log.log')
        file_handler.setLevel(logging.DEBUG)
        if 'log_format' in config:
            formatter = logging.Formatter(config['log_format'])
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # set seed
        self.seed = seeds[config.get(suggests.repeater.TRIAL_INDEX, 0)]
        _ = utils.manual_seed(self.seed, deterministic = False)
        
        # set test domain
        test_room = config.get('test_room', 1)
        
        # load data
        train_set, test_set = data_loader.get_datasets(
                data_dir,
                class_num = self.conf.class_num,
                gram_type = self.conf.gram_type,
                rooms_list = [[i for i in range(self.conf.data_count['room']) if i != test_room], [test_room]],
                slicing = self.conf.slicing,
                slice_length = self.conf.slice_length,
                slice_stride = self.conf.slice_stride)
        self.train_loader = data_loader.get_data_loader(train_set, batch_size = self.conf.batch_size, drop_last = True,
                                                        worker_num = worker_num)
        self.test_loader = data_loader.get_data_loader(test_set, batch_size = self.conf.batch_size, drop_last = True,
                                                       worker_num = worker_num)
        
        # create network
        self.model = network.Network(self.conf.network).to(self.device)
        logger.debug(repr(self.model))
        logger.debug(repr(config))
        
        # create engines
        self.trainer, criteria, optimizers = network.create_trainer(
                trial_id + '.trainer', self.model, self.device,
                self.conf)
        self.train_evaluator = network.create_validator(
                trial_id + '.train_evaluator', self.model, self.device,
                criteria, self.conf)
        self.validator = network.create_validator(
                trial_id + '.validator', self.model, self.device,
                criteria, self.conf)
        
        # monitor the best validation accuracy
        self.best_performance_handler = ignite_handlers.BestPerformance(
                score_function = lambda engine: engine.state.metrics['accuracy'],
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer),
                num_kept = 4)
        self.validator.add_event_handler(ig_engine.Events.COMPLETED, self.best_performance_handler)
        
        # log metrics
        log_train_metrics = ignite_handlers.MetricsLogger(
                file_path = output_dir / self.conf.train_metrics_name,
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer))
        self.train_evaluator.add_event_handler(ig_engine.Events.EPOCH_COMPLETED, log_train_metrics)
        log_validator_metrics = ignite_handlers.MetricsLogger(
                file_path = output_dir / self.conf.validation_metrics_name,
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer))
        self.validator.add_event_handler(ig_engine.Events.EPOCH_COMPLETED, log_validator_metrics)
        
        # Record metric history
        self.train_recorder = ignite_handlers.HistoryRecorder(
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer))
        self.train_evaluator.add_event_handler(ig_engine.Events.COMPLETED, self.train_recorder)
        self.recorder = ignite_handlers.HistoryRecorder(
                metric_names = "all",
                global_step_transform = ig_handlers.global_step_from_engine(self.trainer))
        self.validator.add_event_handler(ig_engine.Events.COMPLETED, self.recorder)
        
        # validate every epoch
        @self.trainer.on(ig_engine.Events.EPOCH_COMPLETED)
        def run_validation(_: ig_engine.Engine):
            """
            Run train evaluator and validation.
    
            :param _: The trainer.
            """
            self.train_evaluator.run(self.train_loader)
            self.validator.run(self.test_loader)
        
        self.iteration_count = 0
    
    def _train(self) -> dict:
        self.iteration_count += 1
        
        # run trainer
        self.trainer.run(self.train_loader, max_epochs = self.epoch_per_iteration * self.iteration_count)
        
        mean_train_metrics = self.train_recorder.get(self.epoch_per_iteration * (self.iteration_count - 1) + 1,
                                                     self.epoch_per_iteration * self.iteration_count + 1, 'mean')
        mean_metrics = self.recorder.get(self.epoch_per_iteration * (self.iteration_count - 1) + 1,
                                         self.epoch_per_iteration * self.iteration_count + 1, 'mean')
        
        return {"mean_accuracy": mean_metrics['accuracy'],
                "mean_loss": mean_metrics['loss_total'],
                "mean_metrics": mean_metrics,
                "train_mean_metrics": mean_train_metrics,
                "best_accuracy": self.best_performance_handler.kept[0].score,
                "best_performance_epoch": self.best_performance_handler.kept[0].epoch,
                "best_metrics": self.best_performance_handler.kept[0].metrics}
    
    def _save(self, checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path
    
    def _restore(self, checkpoint: str):
        self.model.load_state_dict(torch.load(checkpoint))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action = "store_true", help = "Finish quickly for testing")
    parser.add_argument("--data-dir", default = '/data/wzy/WifiEnvironment/gram_data', help = "path to dataset")
    parser.add_argument("--output-dir", default = './tune_results', help = "path to output logs")
    args = parser.parse_args()
    
    # config logging
    output_dir = pathlib.Path(args.output_dir)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        if not os.path.isdir(output_dir):
            raise FileExistsError("Please provide a path to a non-existing or directory as the output directory.")
    log_format = '%(asctime)s: %(levelname)s: %(name)s: %(filename)s: %(funcName)s(): %(lineno)d:\t%(message)s'
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(str(output_dir / 'log.log'), mode = 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    formatter = logging.Formatter(fmt = "[%(levelname)s %(asctime)s] %(name)s: %(message)s",
                                  datefmt = "%m-%d %H:%M:%S", )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level = logging.DEBUG, handlers = [file_handler, stream_handler])
    ax_client.logger.addHandler(file_handler)
    
    # initialize ray
    ray.init()
    
    # trail scheduler
    scheduler = schedulers.AsyncHyperBandScheduler(metric = 'mean_accuracy', mode = "max", grace_period = 3, max_t = 20)
    
    # search algorithm
    client = ax_client.AxClient(enforce_sequential_optimization = False)
    client.create_experiment(
            parameters = hyper_parameters,
            name = "My Experiment",
            objective_name = "best_accuracy",
            parameter_constraints = hyper_parameter_constraints)
    search_alg = ax_suggest.AxSearch(client, mode = "max")
    # It is recommended to not use Repeater with a TrialScheduler. Early termination can negatively affect the
    # average reported metric.
    # search_alg = suggests.Repeater(search_alg, repeat = 2)
    search_alg = suggests.ConcurrencyLimiter(search_alg, max_concurrent = 2)
    
    # run trails
    experiment = tune.Experiment(
            name = "My Experiment",
            run = Trainable,
            stop = {'training_iteration': 5, } if args.smoke_test else {'mean_accuracy': 0.95},
            config = {
                    'epoch_per_iteration': 2 if args.smoke_test else 4, 'batch_size': 10,
                    'test_room': 1,
                    'data_dir': pathlib.Path(args.data_dir)},
            resources_per_trial = {
                    "cpu": 1,
                    "gpu": 1},
            num_samples = 3 if args.smoke_test else 60,
            local_dir = args.output_dir,
            checkpoint_freq = 1 if args.smoke_test else 0,
            checkpoint_at_end = True,
            max_failures = 5)
    analysis = tune.run(
            experiment,
            scheduler = scheduler,
            search_alg = search_alg, )
    
    logging.getLogger('main').critical("Best config is:", analysis.get_best_config(metric = "best_accuracy"))
    print("Best config is:", analysis.get_best_config(metric = "best_accuracy"))
