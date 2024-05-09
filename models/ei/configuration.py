import dataclasses
from typing import Dict, List, Tuple, Union

from configs2 import activity_class_num, data_count, gram_channel_num


@dataclasses.dataclass
class NetworkConfiguration:
    """Network hyper-parameters."""
    activator_type: str = 'relu'  # relu for cnn, soft plus for fc
    feature_size: int = 900  # manually calculated
    domain_feature_size: int = 907  # manually calculated
    feature_extractor: Dict[str, Union[int, List[int], List[Tuple[int, int]], Tuple[int, int]]] = dataclasses.field(
            default_factory = dict(
                    norm_position = 'after_activation',  # not mentioned in paper
                    layer = 3,
                    in_channel_num = gram_channel_num,
                    channel_num = [16, 8, 4],  # not mentioned in paper
                    kernel_size = [(3, 3)] * 3,  # not mentioned in paper
                    padding_size = [(0, 0)] * 3,  # not mentioned in paper, assume no padding
                    pool_kernel_size = [(2, 2)] * 3,
                    pool_stride = [(2, 2)] * 3).copy)  # assume no padding for pooling
    
    classifier: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            layer = 2,
            out_size = [150]).copy)  # not mentioned in paper
    discriminator: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            layer = 2,
            out_size = [150]).copy)  # not mentioned in paper
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

