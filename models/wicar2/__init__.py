"""
Improved WiCAR.
WiCAR: WiFi-based in-Car Activity Recognition with Multi-Adversarial Domain Adaptation
With my CNN and discriminator design.
"""
from .configuration import NetworkConfiguration, LossConfiguration
from .creat_engines import (create_tester, create_trainer, create_validator)
from .network import Network as Network
