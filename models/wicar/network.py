from typing import Optional, Tuple

import torch
from torch import nn as nn

from models.base_common.network import get_activator
from models.domain_adaptation_common.network import GradientReversalFunction
from .configuration import NetworkConfiguration as NetworkConfiguration


class FeatureExtractor(nn.Module):
    """
    Feature extractor.
    
    We use two stacked CNN layers to extract features from the input spectrograms, each followed with a rectified
    linear unit (ReLU) layer as the activation function. Max pooling layers are also applied to reduce the feature
    dimensions.
    """
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters.
        """
        super(FeatureExtractor, self).__init__()
        
        activator = get_activator(conf.activator_type)
        
        self.feature_extractor = nn.Sequential()
        fe_conf = conf.feature_extractor
        
        for i in range(0, fe_conf['layer']):
            if i == 0:
                in_channels = fe_conf['in_channel_num']
            else:
                in_channels = fe_conf['channel_num'][i - 1]
            
            self.feature_extractor.add_module(
                    'conv{no}'.format(no = i),
                    nn.Conv2d(in_channels = in_channels,
                              out_channels = fe_conf['channel_num'][i],
                              kernel_size = fe_conf['kernel_size'][i],
                              padding = fe_conf['padding_size'][i]))
            if fe_conf['norm_position'] == 'before_activation':
                self.feature_extractor.add_module(
                        'bn{no}'.format(no = i),
                        nn.BatchNorm2d(fe_conf['channel_num'][i]))
            self.feature_extractor.add_module(
                    'relu{no}'.format(no = i),
                    activator())
            if fe_conf['norm_position'] == 'after_activation':
                self.feature_extractor.add_module(
                        'bn{no}'.format(no = i),
                        nn.BatchNorm2d(fe_conf['channel_num'][i]))
            self.feature_extractor.add_module(
                    'pool{no}'.format(no = i),
                    nn.MaxPool2d(kernel_size = fe_conf['pool_kernel_size'][i],
                                 stride = fe_conf['pool_stride'][i]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs.
        """
        batch_size, *_ = x.size()
        
        features = self.feature_extractor(x)
        features = features.contiguous().view(batch_size, -1)
        
        return features


class Discriminator(nn.Module):
    """
    Discriminator.
    
    In our model, two fully connect layers with corresponding activation layers are employed to learn the
    discriminative features. At last, a softmax layer is used to map the features to a latent space with the same
    size as the activity label space.
    """
    
    def __init__(self, conf: NetworkConfiguration, output_size: int, feature_size: int = None):
        """
        :param conf: Hyper-parameters.
        :param output_size: The output size.
        
        """
        super(Discriminator, self).__init__()
        
        activator = get_activator(conf.activator_type)
        
        self.discriminator = nn.Sequential()
        disc_conf = conf.discriminator
        if not feature_size:
            feature_size = conf.feature_size
        
        for i in range(0, disc_conf['layer']):
            if i == 0:
                in_size = feature_size
            else:
                in_size = disc_conf['out_size'][i - 1]
            if i == disc_conf['layer'] - 1:
                out_size = output_size
            else:
                out_size = disc_conf['out_size'][i]
            
            self.discriminator.add_module(
                    'fc{no}'.format(no = i),
                    nn.Linear(in_size, out_size))
            if not i == disc_conf['layer'] - 1 and not i == disc_conf['layer'] - 2 and \
                    disc_conf['norm_position'] == 'before_activation':
                self.discriminator.add_module(  # no normalization for the last two layers
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(out_size))
            if not i == disc_conf['layer'] - 1:  # the last layer doesn't need activator
                self.discriminator.add_module(
                        'relu{no}'.format(no = i),
                        activator())
            if not i == disc_conf['layer'] - 1 and not i == disc_conf['layer'] - 2 and \
                    disc_conf['norm_position'] == 'after_activation':
                self.discriminator.add_module(  # no normalization for the last two layers
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(out_size))
        
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        output = self.discriminator(x)
        return output, self.softmax(output)


class Network(nn.Module):
    """
    Network.
    """
    available_modes = ['train_grl', 'whole', 'base']
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        super(Network, self).__init__()
        
        self.feature_extractor = FeatureExtractor(conf)
        self.classifier = Discriminator(conf, conf.class_num)
        
        self.domain_discriminators = nn.ModuleList()
        for i in range(len(conf.domain_out_size)):
            self.domain_discriminators.append(Discriminator(conf, conf.domain_out_size[i], conf.domain_feature_size))
    
    def forward(self,
                x: torch.Tensor,
                mode: str = 'whole',
                lambda0: Optional[float] = None):
        """
        Forward.
        
        :param x: Inputs.
        :param mode:
        :param lambda0: The weight for domain discriminator loss when using grl.
        :return: Outputs and outputs after softmax.
        """
        if mode not in self.available_modes:
            raise ValueError('Unknown working mode.')
        
        features = self.feature_extractor(x)
        
        pred_labels, softmaxed_pred_labels = self.classifier(features)
        
        if mode == 'base':
            return pred_labels, softmaxed_pred_labels
        
        # discriminate domains
        # Each domain discriminator takes as input the concatenation of the feature representations z from the
        # feature encoder and the label distributions yˆ from the activity predictor
        y1 = softmaxed_pred_labels.detach()
        domain_features_ = torch.cat([features, y1], dim = 1)
        if mode in ['train_grl']:
            # use gradient reversal layer technique
            assert (lambda0 is not None)
            domain_features = GradientReversalFunction.apply(domain_features_, lambda0)
        elif mode in ['whole']:
            domain_features = domain_features_
        else:
            raise RuntimeError("Shouldn't reach here.")
        
        pred_domain_labels_list = list()
        softmaxed_pred_domain_labels_list = list()
        for domain_discriminator in self.domain_discriminators:
            pred_domain_labels, softmaxed_pred_domain_labels = domain_discriminator(domain_features)
            pred_domain_labels_list.append(pred_domain_labels)
            softmaxed_pred_domain_labels_list.append(softmaxed_pred_domain_labels)
        
        if mode in ['train_grl', 'whole']:
            return (pred_labels, softmaxed_pred_labels,
                    *zip(pred_domain_labels_list, softmaxed_pred_domain_labels_list))
        else:
            raise RuntimeError("Shouldn't reach here.")
