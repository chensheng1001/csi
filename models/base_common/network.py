import functools

from torch import nn as nn


def get_activator(activator_type: str, alpha: float = 1., negative_slope: float = 1e-2):
    r"""
    
    :param activator_type: Activator type.
    :param alpha: :math:`\alpha` value for the ELU formulation.
    :param negative_slope: `negative_slope` parameter of LeakyReLU.
    :return: The activation function.
    """
    if activator_type == 'elu':
        activator = functools.partial(nn.ELU, alpha = alpha)
    elif activator_type == 'leaky_relu':
        activator = functools.partial(nn.LeakyReLU, negative_slope = negative_slope)
    elif activator_type == 'relu':
        activator = nn.ReLU
    else:
        raise ValueError("Unknown activator type {type} set.".format(type = activator_type))
    return activator
