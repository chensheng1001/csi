from models.mine.network import Network as FullNetwork
from .configuration import NetworkConfiguration


class Network(FullNetwork):
    """
    Network.
    """
    available_modes = ['train_grl', 'whole', 'base']
    
    def __init__(self, conf: NetworkConfiguration):
        """
        :param conf: Hyper-parameters for the network.
        """
        assert len(conf.domain_out_size) == 1
        super(Network, self).__init__(conf)
