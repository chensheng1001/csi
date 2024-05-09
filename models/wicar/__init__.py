"""
WiCAR: WiFi-based in-Car Activity Recognition with Multi-Adversarial Domain Adaptation
"""
from .configuration import LossConfiguration, NetworkConfiguration
from .creat_engines import (create_tester, create_trainer, create_validator)
from .network import Network as Network
