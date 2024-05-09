"""
My model with only discriminator, no label smoothening, no smoothness regularizer, trained using grl
"""
from .configuration import NetworkConfiguration
from .creat_engines import (create_tester, create_trainer, create_validator)
from .network import Network as Network
