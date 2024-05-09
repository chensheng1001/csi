from typing import Optional

import torch
from torch import nn as nn

from models.cnn.network import FeatureExtractor
from models.domain_adaptation_common.network import GradientReversalFunction
from models.wicar.network import Discriminator
from .configuration import NetworkConfiguration as NetworkConfiguration


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
