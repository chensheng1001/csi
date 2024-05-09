import dataclasses
from typing import Dict, List, Tuple, Union

from configs2 import activity_class_num, data_count, gram_channel_num


@dataclasses.dataclass
class NetworkConfiguration:
    """Network hyper-parameters."""
    activator_type: str = 'leaky_relu'  # soft plus for fc
    activator_negative_slope: float = 1e-2
    feature_size: int = 64
    domain_feature_size: int = feature_size + activity_class_num
    feature_extractor: Dict[str, Union[int, List[int], List[Tuple[int, int]], Tuple[int, int]]] = dataclasses.field(
            default_factory = dict(
                    layer = 6,
                    in_channel_num = gram_channel_num,
                    channel_num = [32, 64, 96, 96, 128, 128],
                    pool_kernel_size = [(3, 3)] * 6,
                    pool_stride = [(2, 2)] * 6,
                    global_pool_channel_num = feature_size,
                    global_pool_out_size = (1, 1)).copy)
    
    classifier: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            layer = 3,
            out_size = [256, 256, activity_class_num]).copy)
    discriminator: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            layer = 3,
            out_size = [256, 256, activity_class_num]).copy)
    domain_out_size: List[int] = dataclasses.field(default_factory = list(
            [data_count['room']]).copy)
    class_num: int = activity_class_num


@dataclasses.dataclass
class LossConfiguration:
    """Loss configuration."""
    # weight for domain discriminator loss
    lambda_disc: float = 0.2  # not mentioned in paper
    lambda_confidence: float = 0.1  # not mentioned in paper
    lambda_smoothness: float = 0.1  # not mentioned in paper
    smoothness_epsilon: float = 0.01  # not mentioned in paper

