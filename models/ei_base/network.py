import torch
from torch import nn as nn

from models.ei.network import Classifier, FeatureExtractor
from .configuration import NetworkConfiguration as NetworkConfiguration


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
        self.classifier = Classifier(conf, conf.class_num)
    
    def forward(self,
                x: torch.Tensor):
        """
        Forward.
        
        :param x: Inputs.
        :return: Outputs and outputs after softmax.
        """
        
        features = self.feature_extractor(x)
        
        pred_labels, softmaxed_pred_labels, classifier_features = self.classifier(features)
        
        return pred_labels, softmaxed_pred_labels
