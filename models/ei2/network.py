import torch
from torch import nn as nn

from models.cnn.network import FeatureExtractor
from models.ei.network import Classifier, Discriminator
from .configuration import NetworkConfiguration as NetworkConfiguration


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
