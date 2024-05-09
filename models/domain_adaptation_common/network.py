import math

import torch
from torch import autograd as auto_grad, nn as nn


class GradientReversalFunction(auto_grad.Function):
    """
    The so-called gradient reversal layer
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 0.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.neg().mul(ctx.alpha)
        return grad_input, None  # same number as forward arguments.


class RandomLayer(nn.Module):
    """
    Same as CDAN
    """
    
    def __init__(self, input_size_list: list, output_size: int = 1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_size_list)
        assert self.input_num == 2
        self.output_size = output_size
        
        random_matrix0 = torch.randn(input_size_list[0], output_size)
        self.register_buffer('random_matrix0', random_matrix0)
        random_matrix1 = torch.randn(input_size_list[1], output_size)
        self.register_buffer('random_matrix1', random_matrix1)
    
    def forward(self, input0: torch.Tensor, input1: torch.Tensor) -> torch.Tensor:
        """
        Forward.
        
        :param input0: input0
        :param input1: input0
        :return: Randomized product.
        """
        randomized0 = torch.mm(input0, self.random_matrix0)
        randomized1 = torch.mm(input1, self.random_matrix1)
        output = randomized0 / math.pow(float(self.output_size), 1.0 / self.input_num)
        output = torch.mul(output, randomized1)
        
        return output
