from typing import Tuple

import torch
from torch import nn as nn

from models.base_common.network import get_activator
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


class Classifier(nn.Module):
    """
    Classifier.
    
    A fully-connected layer followed by an activation function is used to learn the representation Vi of Xi.
    Moreover, a softmax layer is used to obtain the probability vector of activities.
    """
    
    def __init__(self, conf: NetworkConfiguration, output_size: int, feature_size: int = None):
        """
        :param conf: Hyper-parameters.
        :param output_size: The output size.
        
        """
        super(Classifier, self).__init__()
        
        activator = nn.Softplus
        
        self.classifier = nn.Sequential()
        clas_conf = conf.classifier
        if not feature_size:
            feature_size = conf.feature_size
        
        for i in range(0, clas_conf['layer'] - 1):
            if i == 0:
                in_size = feature_size
            else:
                in_size = clas_conf['out_size'][i - 1]
            out_size = clas_conf['out_size'][i]
            
            self.classifier.add_module(
                    'fc{no}'.format(no = i),
                    nn.Linear(in_size, out_size))
            
            self.classifier.add_module(
                    'softplus{no}'.format(no = i),
                    activator())
        
        self.softmax_layer = nn.Linear(clas_conf['out_size'][clas_conf['layer'] - 2], output_size)
        
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        features = self.classifier(x)
        output = self.softmax_layer(features)
        return output, self.softmax(output), features


class Discriminator(nn.Module):
    """
    Discriminator.
    
    two fully connected layers with corresponding activation functions are used to project F into domain
    distributions S.
    Ui = Softplus(Wf Fi + bf ).
    Si = Softmax(Wu Ui + bu )
    """
    
    def __init__(self, conf: NetworkConfiguration, output_size: int, feature_size: int = None):
        """
        :param conf: Hyper-parameters.
        :param output_size: The output size.
        
        """
        super(Discriminator, self).__init__()
        
        activator = nn.Softplus
        
        self.discriminator = nn.Sequential()
        disc_conf = conf.discriminator
        if not feature_size:
            feature_size = conf.feature_size
        
        for i in range(0, disc_conf['layer'] - 1):
            if i == 0:
                in_size = feature_size
            else:
                in_size = disc_conf['out_size'][i - 1]
            out_size = disc_conf['out_size'][i]
            
            self.discriminator.add_module(
                    'fc{no}'.format(no = i),
                    nn.Linear(in_size, out_size))
            
            self.discriminator.add_module(
                    'softplus{no}'.format(no = i),
                    activator())
        
        self.softmax_layer = nn.Linear(disc_conf['out_size'][disc_conf['layer'] - 2], output_size)
        
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        output = self.discriminator(x)
        output = self.softmax_layer(output)
        return output, self.softmax(output)


class Network(nn.Module):
    """
    Network.
    """
    available_modes = ['train_ad_domain', 'whole', 'base', 'domain']
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        assert len(conf.domain_out_size) == 1
        super(Network, self).__init__()
        
        self.feature_extractor = FeatureExtractor(conf)
        self.classifier = Classifier(conf, conf.class_num)
        
        self.domain_discriminators = nn.ModuleList()
        for i in range(len(conf.domain_out_size)):
            self.domain_discriminators.append(Discriminator(conf, conf.domain_out_size[i], conf.domain_feature_size))
    
    def forward(self,
                x: torch.Tensor,
                mode: str = 'whole'):
        """
        Forward.
        
        :param x: Inputs.
        :param mode:
        :return: Outputs and outputs after softmax.
        """
        if mode not in self.available_modes:
            raise ValueError('Unknown working mode.')
        
        features = self.feature_extractor(x)
        
        pred_labels, softmaxed_pred_labels, classifier_features = self.classifier(features)
        
        if mode == 'base':
            return pred_labels, softmaxed_pred_labels, classifier_features
        
        # discriminate domains
        # concatenate the output matrix of feature extractor (i.e., Z) and the prediction matrix yˆ
        y1 = softmaxed_pred_labels.detach()
        domain_features_ = torch.cat([features, y1], dim = 1)
        
        if mode in ['train_ad_domain']:
            # real adversarial training
            domain_features = domain_features_.detach()
        elif mode in ['whole', 'domain']:
            domain_features = domain_features_
        else:
            raise RuntimeError("Shouldn't reach here.")
        
        pred_domain_labels, softmaxed_pred_domain_labels = self.domain_discriminators[0](domain_features)
        
        if mode in ['whole']:
            return pred_labels, softmaxed_pred_labels, pred_domain_labels, softmaxed_pred_domain_labels, \
                   classifier_features
        elif mode in ['train_ad_domain', 'domain']:
            return pred_domain_labels, softmaxed_pred_domain_labels,
        else:
            raise RuntimeError("Shouldn't reach here.")
