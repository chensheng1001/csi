import dataclasses
from typing import Dict, List, Tuple, Union

from configs2 import activity_class_num, data_count, gram_channel_num


@dataclasses.dataclass
class NetworkConfiguration:
    """Network hyper-parameters."""
    activator_type: str = 'relu'
    feature_size: int = 4320  # manually calculated
    domain_feature_size: int = 4327  # manually calculated
    feature_extractor: Dict[str, Union[int, List[int], List[Tuple[int, int]], Tuple[int, int]]] = dataclasses.field(
            default_factory = dict(
                    norm_position = 'after_activation',  # not mentioned in paper
                    layer = 2,
                    in_channel_num = gram_channel_num,
                    channel_num = [16, 4],  # not mentioned in paper
                    kernel_size = [(5, 5)] * 2,
                    padding_size = [(0, 0)] * 2,  # not mentioned in paper, assume no padding
                    pool_kernel_size = [(2, 2)] * 2,
                    pool_stride = [(2, 2)] * 2).copy)  # assume no padding for pooling
    discriminator: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            norm_position = 'after_activation',  # not mentioned in paper
            layer = 3,
            out_size = [150, 80]).copy)
    # the paper says two layer with 150, 80 and a softmax layer, so we assume a third layer with class_num hidden units.
    domain_out_size: List[int] = dataclasses.field(default_factory = list(
            [data_count['room'], data_count['user']]).copy)
    class_num: int = activity_class_num


@dataclasses.dataclass
class LossConfiguration:
    """Loss configuration."""
    # weight for domain discriminator loss
    lambda_disc: float = 0.15  # not mentioned in paper
