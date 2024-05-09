"""
Improved EI
With my CNN and discriminator design.
"""
from .configuration import NetworkConfiguration, LossConfiguration
from .creat_engines import (create_tester, create_trainer, create_validator)
from .network import Network as Network
