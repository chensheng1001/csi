from typing import Tuple

import torch
from torch import nn as nn

from models.base_common.network import get_activator
from .configuration import NetworkConfiguration as NetworkConfiguration


class FeatureExtractor(nn.Module):
    """
    Feature extractor
    """
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        super(FeatureExtractor, self).__init__()
        
        activator = get_activator(conf.activator_type, negative_slope = conf.activator_negative_slope)
        
        self.feature_extractor = nn.Sequential()
        fe_conf = conf.feature_extractor
        
        # convolution layers
        for i in range(0, fe_conf['layer']):
            if i == 0:
                in_channels = fe_conf['in_channel_num']
            else:
                in_channels = fe_conf['channel_num'][i - 1]
            
            self.feature_extractor.add_module(
                    'conv{no}'.format(no = i),
                    nn.Conv2d(in_channels = in_channels,
                              out_channels = fe_conf['channel_num'][i],
                              kernel_size = (3, 3),
                              # stride = 1,
                              padding = (1, 1)))
            self.feature_extractor.add_module(
                    '{type}{no}'.format(type = conf.activator_type, no = i),
                    activator())
            self.feature_extractor.add_module(
                    'bn{no}'.format(no = i),
                    nn.BatchNorm2d(fe_conf['channel_num'][i]))
            
            # pooling every two layer
            if i % 2 == 1:
                self.feature_extractor.add_module(
                        'pool{no}'.format(no = i),
                        nn.MaxPool2d(kernel_size = fe_conf['pool_kernel_size'][i],
                                     stride = fe_conf['pool_stride'][i],
                                     padding = (1, 1)))
        
        # global average pooling
        # use a Network in Network block before global pooling function
        self.feature_extractor.add_module(
                'nin{no}_conv'.format(no = 0),
                nn.Conv2d(in_channels = fe_conf['channel_num'][fe_conf['layer'] - 1],
                          out_channels = fe_conf['global_pool_channel_num'],
                          kernel_size = (3, 3),
                          # stride = 1,
                          padding = (1, 1)))
        self.feature_extractor.add_module(
                'nin{no}_{type}{no1}'.format(type = conf.activator_type, no = 0, no1 = 0),
                activator())
        self.feature_extractor.add_module(
                'nin{no}_bn'.format(no = 0),
                nn.BatchNorm2d(fe_conf['global_pool_channel_num']))
        for i in range(0, 2):
            self.feature_extractor.add_module(
                    'nin{no}_conv1{no1}'.format(no = 0, no1 = i),
                    nn.Conv2d(in_channels = fe_conf['global_pool_channel_num'],
                              out_channels = fe_conf['global_pool_channel_num'],
                              kernel_size = (1, 1),
                              padding = 0))
            self.feature_extractor.add_module(
                    'nin{no}_{type}{no1}'.format(type = conf.activator_type, no = 0, no1 = i + 1),
                    activator())
            self.feature_extractor.add_module(
                    'nin{no}_bn{no1}'.format(no = 0, no1 = i + 1),
                    nn.BatchNorm2d(fe_conf['global_pool_channel_num']))
        self.feature_extractor.add_module(
                'global_avg_pool',
                nn.AdaptiveAvgPool2d(fe_conf['global_pool_out_size']))
    
    def forward(self,
                x: torch.Tensor):
        """
        Forward.
        
        :param x: Inputs.
        :return: Features.
        """
        batch_size, *_ = x.size()
        
        features = self.feature_extractor(x)
        features = features.contiguous().view(batch_size, -1)
        
        return features


class Classifier(nn.Module):
    """
    Classifier.
    """
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters.
        """
        super(Classifier, self).__init__()
        
        activator = get_activator(conf.activator_type, negative_slope = conf.activator_negative_slope)
        
        self.classifier = nn.Sequential()
        cl_conf = conf.classifier
        
        for i in range(0, cl_conf['layer']):
            if i == 0:
                in_features = conf.feature_size
            else:
                in_features = cl_conf['out_size'][i - 1]
            
            self.classifier.add_module(
                    "fc{no}".format(no = i),
                    nn.Linear(in_features, cl_conf['out_size'][i]))
            if i != cl_conf['layer'] - 1:  # the last layer doesn't need activator
                self.classifier.add_module(
                        '{type}{no}'.format(type = conf.activator_type, no = i),
                        activator())
            if not i == cl_conf['layer'] - 1 and not i == cl_conf['layer'] - 2:
                self.classifier.add_module(  # no normalization for the last two layers
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(cl_conf['out_size'][i]))
        
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward.

        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        output = self.classifier(x)
        return output, self.softmax(output)


class Network(nn.Module):
    """
    Network.
    """
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        super(Network, self).__init__()
        
        self.feature_extractor = FeatureExtractor(conf)
        self.classifier = Classifier(conf)
    
    def forward(self, x: torch.Tensor):
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        features = self.feature_extractor(x)
        pred_labels, softmaxed_pred_labels = self.classifier(features)
        
        return pred_labels, softmaxed_pred_labels
