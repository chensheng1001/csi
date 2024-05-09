import pathlib
from typing import Dict

data_dir = pathlib.Path('/workspace/data/gram_data_widar/all_gram')
output_dir = pathlib.Path('/workspace/output')
older_output_dir = pathlib.Path('/data/wzy/WifiEnvironment/out2')

# data_count: Dict[str, int] = {'room': 4, 'user': 5, 'position': 4, 'activity': 9, 'repetition': 20}
# activity_class_num: int = 7

# data_count: Dict[str, int] = {'room': 4, 'user': 8, 'positions':24, 'repetition': 10}
# activity_class_num: int = 8


data_count: Dict[str, int] = {'room': 4, 'user': 8, 'positions':24, 'repetition': 10}
activity_class_num: int = 8

# 10 ms per sampling point
time_resolution: int = 10
# pca_k
gram_channel_num = 12

model_comparison_epochs = 100
model_comparison_seeds = [80126033, 76847987, 45095068]  # , 91525521, 26114759]
hparams_comparison_seeds = [26360154, 83182821, 39315480]
