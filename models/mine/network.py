from typing import Optional, Tuple

import torch
from torch import nn as nn

from models.base_common.network import get_activator
from models.domain_adaptation_common.network import GradientReversalFunction, RandomLayer
from .configuration import NetworkConfiguration


class FeatureExtractor(nn.Module):
    """
    Feature extractor
    """
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters.
        """
        super(FeatureExtractor, self).__init__()
        
        activator = get_activator(conf.activator_type, conf.activator_alpha, conf.activator_negative_slope)
        
        # local extractor
        self.local_extractor = nn.Sequential()
        le_conf = conf.local_extractor
        
        # convolution layers
        for i in range(0, le_conf['layer']):
            if i == 0:
                in_channels = le_conf['in_channel_num']
            else:
                in_channels = le_conf['channel_num'][i - 1]
            
            self.local_extractor.add_module(
                    'conv{no}'.format(no = i),
                    nn.Conv2d(in_channels = in_channels,
                              out_channels = le_conf['channel_num'][i],
                              kernel_size = (3, 3),
                              # stride = 1,
                              padding = (1, 1)))
            if le_conf['norm_position'] == 'before_activation':
                self.local_extractor.add_module(
                        'bn{no}'.format(no = i),
                        nn.BatchNorm2d(le_conf['channel_num'][i]))
            self.local_extractor.add_module(
                    '{type}{no}'.format(type = conf.activator_type, no = i),
                    activator())
            if le_conf['norm_position'] == 'after_activation':
                self.local_extractor.add_module(
                        'bn{no}'.format(no = i),
                        nn.BatchNorm2d(le_conf['channel_num'][i]))
            
            # pooling every two layer
            if i % 2 == 1:
                self.local_extractor.add_module(
                        'pool{no}'.format(no = i),
                        nn.MaxPool2d(kernel_size = le_conf['pool_kernel_size'][i],
                                     stride = le_conf['pool_stride'][i],
                                     padding = (1, 1)))
        # (batch_size * seq_len, channel_num[-1], ?, ?)
        
        # global average pooling
        # use a Network in Network block before global pooling function
        self.local_extractor.add_module(
                'nin{no}_conv'.format(no = 0),
                nn.Conv2d(in_channels = le_conf['channel_num'][le_conf['layer'] - 1],
                          out_channels = le_conf['global_pool_channel_num'],
                          kernel_size = (3, 3),
                          # stride = 1,
                          padding = (1, 1)))
        if le_conf['norm_position'] == 'before_activation':
            self.local_extractor.add_module(
                    'nin{no}_bn'.format(no = 0),
                    nn.BatchNorm2d(le_conf['global_pool_channel_num']))
        self.local_extractor.add_module(
                'nin{no}_{type}{no1}'.format(type = conf.activator_type, no = 0, no1 = 0),
                activator())
        if le_conf['norm_position'] == 'after_activation':
            self.local_extractor.add_module(
                    'nin{no}_bn'.format(no = 0),
                    nn.BatchNorm2d(le_conf['global_pool_channel_num']))
        for i in range(0, 2):
            self.local_extractor.add_module(
                    'nin{no}_conv1{no1}'.format(no = 0, no1 = i),
                    nn.Conv2d(in_channels = le_conf['global_pool_channel_num'],
                              out_channels = le_conf['global_pool_channel_num'],
                              kernel_size = (1, 1),
                              # stride = 1,
                              padding = 0))
            if le_conf['norm_position'] == 'before_activation':
                self.local_extractor.add_module(
                        'nin{no}_bn{no1}'.format(no = 0, no1 = i + 1),
                        nn.BatchNorm2d(le_conf['global_pool_channel_num']))
            self.local_extractor.add_module(
                    'nin{no}_{type}{no1}'.format(type = conf.activator_type, no = 0, no1 = i + 1),
                    activator())
            if le_conf['norm_position'] == 'after_activation':
                self.local_extractor.add_module(
                        'nin{no}_bn{no1}'.format(no = 0, no1 = i + 1),
                        nn.BatchNorm2d(le_conf['global_pool_channel_num']))
        self.local_extractor.add_module(
                'global_avg_pool',
                nn.AdaptiveAvgPool2d(le_conf['global_pool_out_size']))
        # (batch_size * seq_len, global_pool_channel_num, global_pool_out_size[0], global_pool_out_size[1])
        
        # temporal extractor
        te_conf = conf.temporal_extractor
        bidirectional = (te_conf['bidirectional'] is True)
        self.temporal_extractor = nn.Sequential()
        # todo learn initial states
        if te_conf['type'] == 'lstm':
            self.temporal_extractor.add_module(
                    'lstm',
                    nn.LSTM(conf.local_feature_size, te_conf['hidden_size'], num_layers = te_conf['layer'],
                            dropout = te_conf['dropout'], bidirectional = bidirectional, batch_first = True))
        elif te_conf['type'] == 'gru':
            self.temporal_extractor.add_module(
                    'gru',
                    nn.GRU(conf.local_feature_size, te_conf['hidden_size'], num_layers = te_conf['layer'],
                           dropout = te_conf['dropout'], bidirectional = bidirectional, batch_first = True))
        else:
            raise ValueError("Unknown rnn type {type}.".format(type = te_conf['type']))
        if te_conf['norm_position'] == 'yes':
            self.temporal_extractor_norm = nn.BatchNorm1d(conf.feature_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs.
        """
        batch_size, seq_len, channel_num, row_num, col_num = x.size()
        
        x_reshape = x.contiguous().view(batch_size * seq_len, channel_num, row_num, col_num)
        features = self.local_extractor(x_reshape)
        # (batch_size * seq_len, global_pool_channel_num, global_pool_out_size[0], global_pool_out_size[1])
        features = features.contiguous().view(batch_size * seq_len, -1)
        # (batch_size * seq_len, local_feature_size)
        
        features = features.contiguous().view(batch_size, seq_len, -1)
        # (batch_size, seq_len, local_feature_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        output, _ = self.temporal_extractor(features)
        # (batch_size, seq_len, num_directions * hidden_size)
        features = output[:, -1, :]
        if self.temporal_extractor_norm:
            features = self.temporal_extractor_norm(features)
        # (batch_size, num_directions * hidden_size)
        
        return features


class Classifier(nn.Module):
    """
    Classifier.
    """
    
    def __init__(self, conf: NetworkConfiguration, output_size: int):
        """
        :param conf: Hyper-parameters.
        :param output_size: The output size.
        
        """
        super(Classifier, self).__init__()
        
        activator = get_activator(conf.activator_type, conf.activator_alpha, conf.activator_negative_slope)
        
        self.classifier = nn.Sequential()
        cl_conf = conf.classifier
        
        # todo attention
        for i in range(0, cl_conf['layer']):
            if i == 0:
                in_size = conf.feature_size
            else:
                in_size = cl_conf['out_size'][i - 1]
            if i == cl_conf['layer'] - 1:
                out_size = output_size
            else:
                out_size = cl_conf['out_size'][i]
            
            self.classifier.add_module(
                    "fc{no}".format(no = i),
                    nn.Linear(in_size, out_size))
            if not i == cl_conf['layer'] - 1 and not i == cl_conf['layer'] - 2 and \
                    cl_conf['norm_position'] == 'before_activation':
                self.classifier.add_module(  # no normalization for the last two layers
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(out_size))
            if i != cl_conf['layer'] - 1:  # the last layer doesn't need activator
                self.classifier.add_module(
                        '{type}{no}'.format(type = conf.activator_type, no = i),
                        activator())
            if not i == cl_conf['layer'] - 1 and not i == cl_conf['layer'] - 2 and \
                    cl_conf['norm_position'] == 'after_activation':
                self.classifier.add_module(
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(out_size))
        
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        output = self.classifier(x)
        return output, self.softmax(output)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator.
    """
    
    def __init__(self, conf: NetworkConfiguration, output_size: int):
        """
        :param conf: Hyper-parameters.
        :param output_size: The output size.
        
        """
        super(DomainDiscriminator, self).__init__()
        
        activator = get_activator(conf.activator_type, conf.activator_alpha, conf.activator_negative_slope)
        
        self.discriminator = nn.Sequential()
        dd_conf = conf.domain_discriminator
        
        for i in range(0, dd_conf['layer']):
            if i == 0:
                in_size = conf.domain_feature_size
            else:
                in_size = dd_conf['out_size'][i - 1]
            if i == dd_conf['layer'] - 1:
                out_size = output_size
            else:
                out_size = dd_conf['out_size'][i]
            
            self.discriminator.add_module(
                    'fc{no}'.format(no = i),
                    nn.Linear(in_size, out_size))
            if not i == dd_conf['layer'] - 1 and not i == dd_conf['layer'] - 2 and \
                    dd_conf['norm_position'] == 'before_activation':
                self.discriminator.add_module(  # no normalization for the last two layers
                        'bn{no}'.format(no = i),
                        nn.BatchNorm1d(out_size))
            if not i == dd_conf['layer'] - 1:  # the last layer doesn't need activator
                self.discriminator.add_module(
                        '{type}{no}'.format(type = conf.activator_type, no = i),
                        activator())
            if not i == dd_conf['layer'] - 1 and not i == dd_conf['layer'] - 2 and \
                    dd_conf['norm_position'] == 'after_activation':
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
    available_modes = ['train_grl', 'train_ad_domain', 'whole', 'base', 'domain']
    available_domain_feature_types = ['joint_distribution', 'margin_distribution', 'concatenation']
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        assert len(conf.domain_out_size) == 1
        super(Network, self).__init__()
        
        if conf.domain_feature_type in self.available_domain_feature_types:
            self.domain_feature_type = conf.domain_feature_type
        else:
            raise ValueError('Unknown domain feature type.')
        
        self.feature_extractor = FeatureExtractor(conf)
        self.classifier = Classifier(conf, conf.class_num)
        
        if self.domain_feature_type == 'joint_distribution':
            self.random_layer = RandomLayer([conf.feature_size, conf.class_num], conf.feature_size)
        
        self.domain_discriminators = nn.ModuleList()
        for i in range(len(conf.domain_out_size)):
            self.domain_discriminators.append(DomainDiscriminator(conf, conf.domain_out_size[i]))
    
    def forward(self,
                x: torch.Tensor, y: Optional[torch.Tensor] = None,
                mode: str = 'whole',
                lambda0: Optional[float] = None):
        """
        Forward.
        
        :param x: Inputs.
        :param y: True one-hot class labels.
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
        
        # use predicted class label distributions for unlabeled data
        y1 = y.detach() if (y is not None) else softmaxed_pred_labels.detach()
        # # always use predicted class label distributions
        # y1 = softmaxed_pred_labels.detach()
        if self.domain_feature_type == 'joint_distribution':
            # Randomized Multi-linear Conditioning from CDAN
            domain_features_ = self.random_layer(features, y1)
        elif self.domain_feature_type == 'concatenation':
            domain_features_ = torch.cat([features, y1], dim = 1)
        elif self.domain_feature_type == 'margin_distribution':
            domain_features_ = features
        else:
            raise RuntimeError("Shouldn't reach here.")
        
        # discriminate domains
        if mode == 'train_grl':
            # use gradient reversal layer technique
            assert (lambda0 is not None)
            domain_features = GradientReversalFunction.apply(domain_features_, lambda0)
        elif mode == 'train_ad_domain':
            # real adversarial training
            domain_features = domain_features_.detach()
        elif mode in ['whole', 'domain']:
            domain_features = domain_features_
        else:
            raise RuntimeError("Shouldn't reach here.")
        
        # pred_domain_labels_list = list()
        # softmaxed_pred_domain_labels_list = list()
        # for domain_discriminator in self.domain_discriminators:
        #     pred_domain_labels, softmaxed_pred_domain_labels = domain_discriminator(domain_features)
        #     pred_domain_labels_list.append(pred_domain_labels)
        #     softmaxed_pred_domain_labels_list.append(softmaxed_pred_domain_labels)
        #
        # tmp = []
        # for pred_domain_labels, softmaxed_pred_domain_labels in zip(pred_domain_labels_list,
        #                                                             softmaxed_pred_domain_labels_list):
        #     tmp.append(pred_domain_labels)
        #     tmp.append(softmaxed_pred_domain_labels)

        pred_domain_labels, softmaxed_pred_domain_labels = self.domain_discriminators[0](domain_features)
        
        if mode in ['train_grl', 'whole']:
            return pred_labels, softmaxed_pred_labels, pred_domain_labels, softmaxed_pred_domain_labels, features
        elif mode in ['train_ad_domain', 'domain']:
            return pred_domain_labels, softmaxed_pred_domain_labels,
        else:
            raise RuntimeError("Shouldn't reach here.")
