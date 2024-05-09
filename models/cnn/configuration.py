import dataclasses
from typing import Dict, List, Tuple, Union

from configs2 import activity_class_num, gram_channel_num


@dataclasses.dataclass
class NetworkConfiguration:
    """Network hyper-parameters."""
    activator_type: str = 'leaky_relu'
    activator_negative_slope: float = 1e-2
    feature_size: int = 64
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
    class_num: int = activity_class_num
