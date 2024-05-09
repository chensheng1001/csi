import copy
import dataclasses
import pathlib
from typing import Dict, List, Tuple, Union

from configs2 import activity_class_num, data_count, gram_channel_num, time_resolution, data_dir, output_dir



@dataclasses.dataclass
class Loss:
    """Loss configuration."""
    # the loss function used for classifier, cross_entropy/cos
    classifier_type: str = 'cross_entropy'
    # using label smoothen (when Cross Entropy loss function) for classifier, 0.0/0.1/0.2
    classifier_label_smoothen: bool = True
    classifier_label_smooth_factor: float = 0.2
    # using label smoofthen for discriminator, 0.0/0.1/0.2
    discriminator_label_smoothen: bool = True
    discriminator_label_smooth_factor: float = 0.2
    # weight for domain discriminator loss, 0.1~1.0
    lambda_disc: float = 0.2
    # weight for domain discriminator loss in unlabelled fine tune, 0.1~0.9
    lambda_disc_fine_tune: float = 0.2
    # weight for feature space smoothness regularization, 0.0/0.1/0.2/.../0.9
    lambda_smooth: float = 0.1
    # feature space smoothness regularization
    regularization_smooth: bool = True
    # epsilon, 1e-5~1e-2
    smooth_eta: float = 0.0045
    # 5e-3~1
    smooth_alpha: float = 0.5


# hyper-parameters search space
loss_hyper_parameters = [
        {'name': 'classifier_label_smoothing',
         # "type": "choice",
         "values": [0.0, 0.1, 0.2],  # 0.0 means no smoothing
         'value_type': 'float'},
        {'name': 'lambda_disc',
         "type": "range",
         "bounds": [0.1, 0.4],  # 0.9],
         'value_type': 'float'},
        {'name': 'lambda_smooth',
         "type": "choice",
         "values": [0.0, 0.1, 0.2, 0.3, 0.4],  # 0.5, 0.6, 0.7, 0.8, 0.9],  # 0.0 means no regularization
         'value_type': 'float'},
        {'name': 'smooth_eta',
         "type": "range",
         "bounds": [0.00001, 0.01],
         'value_type': 'float'},
        {'name': 'smooth_alpha',
         "type": "range",
         "bounds": [0.005, 1.0],
         'value_type': 'float'}]
loss_hyper_parameters_2 = [
        {'name': 'classifier_label_smoothing',
         "type": "range",
         "bounds": [0.0, 0.2],
         'value_type': 'float'},
        {'name': 'lambda_disc',
         "type": "range",
         "bounds": [0.0, 0.5],
         'value_type': 'float'},
        {'name': 'lambda_smooth',
         "type": "range",
         "bounds": [0.0, 0.5],
         'value_type': 'float'}]
loss_hyper_parameters_4 = [
        {'name': 'classifier_type',
         "type": "choice",
         "values": ['cross_entropy', 'cos'],
         'value_type': 'str'},
        {'name': 'classifier_label_smoothing',
         "type": "choice",
         "values": [0.0, 0.1, 0.2],
         'value_type': 'float'},
        {'name': 'discriminator_label_smoothing',
         "type": "choice",
         "values": [0.0, 0.1, 0.2],
         'value_type': 'float'}]
loss_hyper_parameters_n = [
        {'name': 'classifier_label_smoothing',
         "type": "choice",
         "values": [0.1, 0.2],
         'value_type': 'float'},
        {'name': 'discriminator_label_smoothing',
         "type": "choice",
         "values": [0.1, 0.2],
         'value_type': 'float'},
        {'name': 'lambda_disc',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4, 0.5],
         'value_type': 'float'},
        {'name': 'lambda_smooth',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4, 0.5],
         'value_type': 'float'},
        {'name': 'smooth_eta',
         "type": "range",
         "bounds": [0.00001, 0.01],
         'value_type': 'float'},
        {'name': 'smooth_alpha',
         "type": "range",
         "bounds": [0.005, 1.0],
         'value_type': 'float'}]
loss_hyper_parameters_5 = [
        {'name': 'lambda_disc',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4],
         'value_type': 'float'},
        {'name': 'lambda_smooth',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4],
         'value_type': 'float'}]
loss_hyper_parameters_6 = [
        {'name': 'lambda_disc',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4],
         'value_type': 'float'},
        {'name': 'lambda_smooth',
         "type": "choice",
         "values": [0.1, 0.2, 0.3, 0.4],
         'value_type': 'float'},
        {'name': 'smooth_eta',
         "type": "range",
         "bounds": [0.00001, 0.01],
         'value_type': 'float'},
        {'name': 'smooth_alpha',
         "type": "range",
         "bounds": [0.005, 1.0],
         'value_type': 'float'}]


@dataclasses.dataclass
class Optimizer:
    """Optimizer configuration."""
    type: str = 'adam'
    # learning rate, 1e-5~2e-3
    # learning_rate: float = 3e-4
    learning_rate: float = 3e-3
    # epsilon, 1e-8~1e-5
    eps: float = 1e-7
    # beta_1 for Adam, 0.5~0.99
    beta_1: float = 0.9
    # beta_2 for Adam, 0.9~0.9999
    beta_2: float = 0.99
    # GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    t_t_u_r = True
    # learning rate, 1e-5~2e-3
    learning_rate_discriminator: float = 1e-3


# hyper-parameters search space
optimizer_hyper_parameters = [
        {'name': 'learning_rate',
         "type": "range",
         "bounds": [0.0005, 0.002],  # [0.0001, 0.002],
         'value_type': 'float'},
        {'name': 'beta_1',
         "type": "range",
         "bounds": [0.8, 0.99],  # [0.5, 0.99],
         'value_type': 'float'},
        {'name': 'beta_2',
         "type": "range",
         "bounds": [0.9, 0.9999],
         'value_type': 'float'}]
optimizer_hyper_parameters_n = [
        {'name': 'learning_rate',
         "type": "choice",
         "values": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3],
         'value_type': 'float'},
        {'name': 'eps',
         "type": "choice",
         "values": [1e-8, 1e-7, 1e-6, 1e-5],
         'value_type': 'float'},
        {'name': 'beta_1',
         "type": "choice",
         "values": [0.85, 0.90, 0.95, 0.99],
         'value_type': 'float'},
        {'name': 'beta_2',
         "type": "choice",
         "values": [0.95, 0.99, 0.999, 0.9999],
         'value_type': 'float'}]
optimizer_hyper_parameters_5 = [
        {'name': 'learning_rate',
         "type": "choice",
         "values": [1e-4, 3e-4, 5e-4, 1e-3],
         'value_type': 'float'},
        {'name': 'learning_rate_disc',
         "type": "choice",
         "values": [5e-4, 1e-3, 2e-3],
         'value_type': 'float'}]
optimizer_hyper_parameters_6 = [
        {'name': 'learning_rate',
         "type": "choice",
         "values": [1e-4, 3e-4, 5e-4, 1e-3],
         'value_type': 'float'},
        {'name': 'learning_rate_disc',
         "type": "choice",
         "values": [5e-4, 7e-4, 1e-3, 2e-3],
         'value_type': 'float'}]


@dataclasses.dataclass
class Network:
    """Network hyper-parameters."""
    # activator_type is the activator used by hidden layers, relu/elu/leaky_relu
    # elu might be better than leaky_relu, but much slower
    activator_type: str = 'leaky_relu'
    # the :math:`\alpha` value for the ELU formulation, todo
    activator_alpha: float = 1.0
    # Controls the angle of LeakyReLU's the negative slope, todo
    activator_negative_slope: float = 1e-2
    # the size of features as the output of local extractor as well as the input of temporal extractor,
    # 64/128/256/512
    local_feature_size: int = 64
    # the size of features as the output of feature extractor, same as num_directions * temporal_extractor[
    # hidden_size]
    feature_size: int = local_feature_size * 2
    # the size of features as the input of domain discriminator, feature_size + class_num if concatenated
    domain_feature_size: int = feature_size
    # the hyper-parameters of local feature extractor
    #     norm_position is put normalization before or after activator, before_activation/after_activation/no
    #     layer is the total number of layers, 4/6/8
    #     in_channel_num is the in_channels of the first convolution layer
    #     channel_num is the out_channels of convolution layers, 64... the last one larger than
    #     global_pool_channel_num
    #     kernel_size is the kernel_size of convolution layers, all (3, 3)(height * width)
    #     stride is the stride of convolution layers, all 1
    #     padding is the padding of convolution layers, keep height and width
    #     pool_kernel_size is the kernel_size of pooling, all (2, 2)/(3, 3)(height * width)
    #     pool_stride is the stride of pooling, all (2, 2)(height * width)
    #     global_pool_channel_num is the channels of global average pooling function, local_feature_size
    #     global_pool_out_size is the output_size of global average pooling function, (1, 1)
    local_extractor: Dict[str, Union[int, List[int], List[Tuple[int, int]], Tuple[int, int]]] = dataclasses.field(
            default_factory = dict(
                    norm_position = 'after_activation',
                    layer = 6,
                    in_channel_num = gram_channel_num,
                    channel_num = [32, 64, 96, 96, 128, 128],
                    # kernel_size = [(3, 3)] * 4,
                    # stride = [1] * 4,
                    # padding = [(1, 1)] * 4,
                    pool_kernel_size = [(3, 3)] * 6,
                    pool_stride = [(2, 2)] * 6,
                    global_pool_channel_num = 64,
                    global_pool_out_size = (1, 1)).copy)
    # the hyper-parameters of temporal feature extractor
    #     type is the type of rnn, lstm/gru
    #     norm_position, yes/no
    #     bidirectional is bi-direction or not, True/False
    #     layer is the total number of layers, 1/2
    #     hidden_size is the hidden_size, same as local_feature_size
    #     dropout is the dropout rate, 0.0 - 0.6
    temporal_extractor: Dict[str, Union[str, int, float]] = dataclasses.field(default_factory = dict(
            norm_position = 'yes',
            type = 'gru',
            bidirectional = True,
            layer = 2,
            hidden_size = local_feature_size,
            dropout = 0.3).copy)
    # the hyper-parameters of classifier
    #     norm_position is put normalization before or after activator, before_activation/after_activation/no
    #     layer is the total number of layers, 1/2/3/4
    #     out_size is the out_features of fully-connected layers
    classifier: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            norm_position = 'after_activation',
            layer = 3,
            out_size = [512, 512, activity_class_num]).copy)
    # the hyper-parameters of domain discriminator
    #     norm_position is put normalization before or after activator, before_activation/after_activation/no
    #     layer is the total layers of fc, 1/2/3/4
    #     out_size is the out_features of fully-connected layers
    domain_discriminator: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            norm_position = 'after_activation',
            layer = 3,
            out_size = [512, 512, data_count['room']]).copy)
    domain_out_size: List[int] = dataclasses.field(default_factory = list([data_count['room']]).copy)
    class_num: int = activity_class_num
    # joint_distribution/concatenation/margin_distribution
    domain_feature_type: str = 'joint_distribution'


# hyper-parameters search space
net_hyper_parameters = [
        {'name': 'local_feature_size',
         "type": "choice",
         "values": [64, 128, 256],  # , 512],
         'value_type': 'int'},
        {'name': 'local_extractor_layers',
         "type": "choice",
         "values": [4, 6],  # , 8],
         'value_type': 'int'},
        {'name': 'local_extractor_pool_kernel_size',
         "type": "choice",
         "values": [2, 3],
         'value_type': 'int'},
        {'name': 'temporal_extractor_type',
         "type": "choice",
         "values": ['lstm', 'gru'],
         'value_type': 'str'},
        {'name': 'temporal_extractor_bidirectional',
         "type": "choice",
         "values": [True, False],
         'value_type': 'bool'},
        {'name': 'temporal_extractor_layers',
         "type": "range",
         "bounds": [1, 2],
         'value_type': 'int'},
        {'name': 'temporal_extractor_dropout',
         "type": "range",
         "bounds": [0.0, 0.6],
         'value_type': 'float'},
        {'name': 'classifier_layers',  # domain_discriminator_layers is the same
         "type": "range",
         "bounds": [1, 4],
         'value_type': 'int'}]
net_hyper_parameters_2 = [
        {'name': 'local_feature_size',
         "type": "choice",
         "values": [64, 128],
         'value_type': 'int'},
        {'name': 'local_extractor_layers',
         "type": "choice",
         "values": [4, 6],
         'value_type': 'int'},
        {'name': 'temporal_extractor_type',
         "type": "choice",
         "values": ['lstm', 'gru'],
         'value_type': 'str'},
        {'name': 'temporal_extractor_bidirectional',
         "type": "choice",
         "values": [True, False],
         'value_type': 'bool'},
        {'name': 'temporal_extractor_dropout',
         "type": "range",
         "bounds": [0.0, 0.6],
         'value_type': 'float'},
        {'name': 'classifier_layers',
         "type": "range",
         "bounds": [1, 4],
         'value_type': 'int'}]
net_hyper_parameters_3 = [
        {'name': 'activator_type',
         "type": "choice",
         "values": ['relu', 'elu', 'leaky_relu'],
         'value_type': 'str'},
        {'name': 'local_feature_size',
         "type": "choice",
         "values": [64, 128],
         'value_type': 'int'},
        {'name': 'local_extractor_pool_kernel_size',
         "type": "choice",
         "values": [2, 3],
         'value_type': 'int'},
        {'name': 'temporal_extractor_layers',
         "type": "range",
         "bounds": [1, 2],
         'value_type': 'int'},
        {'name': 'temporal_extractor_dropout',
         "type": "range",
         "bounds": [0.0, 0.5],
         'value_type': 'float'}]
net_hyper_parameters_4 = [
        {'name': 'activator_type',
         "type": "choice",
         "values": ['relu', 'elu', 'leaky_relu'],
         'value_type': 'str'}]
net_hyper_parameters_n = [
        {'name': 'temporal_extractor_dropout',
         "type": "choice",
         "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
         'value_type': 'float'},
        {'name': 'activator_type',
         "type": "choice",
         "values": ['relu', 'elu', 'leaky_relu'],
         'value_type': 'str'}]
net_hyper_parameters_5 = [
        {'name': 'feature_size',
         "type": "choice",
         "values": [128, 256, 512],
         'value_type': 'int'},
        {'name': 'classifier_size0',
         "type": "choice",
         "values": [64, 128, 256, 512],
         'value_type': 'int'},
        {'name': 'classifier_size1',
         "type": "choice",
         "values": [32, 64, 128, 256, 512],
         'value_type': 'int'},
        {'name': 'domain_feature_type',
         "type": "choice",
         "values": ['joint_distribution', 'margin_distribution', 'concatenation'],
         'value_type': 'str'}]


@dataclasses.dataclass
class Configuration:
    """All configurations."""
    # paths
    data_dir: pathlib.Path = copy.deepcopy(data_dir)
    output_dir: pathlib.Path = copy.deepcopy(output_dir)
    
    # logs
    log_name: str = 'log.log'
    # save model during training every checkpoint_interval epoch
    save_models: bool = True
    checkpoint_prefix: str = 'network'
    checkpoint_interval: int = 10
    checkpoint_max_kept: int = 6
    log_to_tensorboard: bool = True
    # save metrics evaluation results to file
    log_metrics_to_file: bool = True
    train_metrics_name: str = 'train.csv'
    validation_metrics_name: str = 'validation.csv' 
    test_metrics_name: str = 'test.csv'
    
    # train / test dataset split ratio, for fine-tune
    train_test_ratio: float = 0.1
    # batch size
    batch_size: int = 16
    # max train epochs
    max_epochs: int = 100
    
    data_count: Dict[str, int] = dataclasses.field(default_factory = data_count.copy)
    class_num: int = activity_class_num
    gram_channel_num: int = gram_channel_num
    
    slicing: bool = True
    # each gram snippet is 400 ms, 40/50/60
    slice_length: int = 40 // time_resolution
    # the stride of gram snippets is 200 ms, 10/20 
    slice_stride: int = 20 // time_resolution
    
    gram_type: str = 'log_stft'  # stft/hht/log_stft/log_hht/ampl
    
    # adversarial training or gradient reversal layer technique
    adversarial_training: bool = True
    # use prediction step to stabilize adversarial training
    stable_adversarial_training: bool = True
    
    loss: Loss = dataclasses.field(default_factory = Loss)
    optimizer: Optimizer = dataclasses.field(default_factory = Optimizer)
    network: Network = dataclasses.field(default_factory = Network)


default_configs = Configuration()
  
# hyper-parameters search space
gram_hyper_parameters = [
        {'name': 'gram_type',
         "type": "choice",
         "values": ['log_stft', 'log_hht', 'stft', 'hht', 'ampl'],
         'value_type': 'str'}]
slicer_hyper_parameters = [
        {'name': 'slice_stride',
         "type": "choice",
         "values": [100 // time_resolution, 200 // time_resolution],
         'value_type': 'int'},
        {'name': 'slice_length',
         "type": "choice",
         "values": [400 // time_resolution, 500 // time_resolution, 600 // time_resolution],
         'value_type': 'int'}]
slicer_hyper_parameters_5 = [
        {'name': 'slice_length',
         "type": "choice",
         "values": [x // time_resolution for x in [300, 400, 600, 800]],
         'value_type': 'int'}]
ad_hyper_parameters = [
        {'naxme': 'stable_adversarial_training',
         "type": "choice",
         "values": [True, False],
         'value_type': 'bool'},
        {'name': 'learning_rate_disc',
         "type": "choice",
         "values": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3],
         'value_type': 'float'}]
ad_hyper_parameters_n = [
        {'name': 'learning_rate_disc',
         "type": "choice",
         "values": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3],
         'value_type': 'float'}]
hyper_parameters = [
        {'name': 'gram_type',
         "type": "choice",
         "values": ['log_stft', 'log_hht'],  # , 'stft', 'hht', 'ampl'],
         'value_type': 'str'},
        {'name': 'slice_stride',
         "type": "choice",
         "values": [100 // time_resolution, 200 // time_resolution],
         'value_type': 'int'},
        *loss_hyper_parameters,
        *optimizer_hyper_parameters,
        *net_hyper_parameters]
hyper_parameters_2 = [
        {'name': 'gram_type',
         "type": "choice",
         "values": ['log_stft', 'log_hht'],
         'value_type': 'str'},
        *loss_hyper_parameters_2,
        *net_hyper_parameters_2]
hyper_parameters_3 = [
        *net_hyper_parameters_3,
        *slicer_hyper_parameters]
hyper_parameters_4 = [
        *loss_hyper_parameters_4,
        *net_hyper_parameters_4]
hyper_parameters_ad = [
        *optimizer_hyper_parameters_n,
        *ad_hyper_parameters]
hyper_parameters_n = [
        *optimizer_hyper_parameters_n,
        *loss_hyper_parameters_n,
        *net_hyper_parameters_n,
        *ad_hyper_parameters_n]
hyper_parameters_5 = [
        *net_hyper_parameters_5,
        *loss_hyper_parameters_5,
        *optimizer_hyper_parameters_5,
        *slicer_hyper_parameters_5]
hyper_parameters_6 = [
        *loss_hyper_parameters_6,
        *optimizer_hyper_parameters_6]
hyper_parameter_constraints = []

if __name__ == '__main__':
    configs = Configuration()
