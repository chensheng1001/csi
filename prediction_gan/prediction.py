from contextlib import contextmanager
from typing import Callable, Iterable, Optional

import torch
import torch.optim as optim


class PredictionStep(optim.Optimizer):
    """
    
    
    :param params: an iterable of :class:`torch.Tensor` s or :class:`dict` s. Specifies what Tensors are
        optimized by the corresponding optimizer.
    """
    
    def __init__(self, params: Iterable):
        super(PredictionStep, self).__init__(params, defaults = dict())
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                
                # state initialization
                if len(state) == 0:
                    state['prev'] = p.data.clone()
                    state['diff'] = p.data - state['prev']
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                
                # Calculate difference and store new params
                state['diff'] = p.data - state['prev']
                state['prev'] = p.data.clone()
        
        return loss
    
    @contextmanager
    def lookahead(self, step: float = 1.0):
        # Do nothing if lookahead step is 0.0
        if step == 0.0:
            yield
            return
        
        if step < 0.0:
            raise ValueError("Lookahead step should not be smaller than zero.")
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                
                # lookahead
                p.data[:] += step * state['diff'][:]
        
        yield
        
        # Roll-back to the original values
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data[:] = state['prev'][:]
