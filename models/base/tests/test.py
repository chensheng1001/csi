import pytorch_model_summary
import torch

from configs import default_configs
from models.base import Network, NetworkConfiguration

if __name__ == '__main__':
    model = Network(NetworkConfiguration())
    sample = torch.rand([8, 37, default_configs.gram_channel_num, 60, default_configs.slice_length])
    print(pytorch_model_summary.summary(model, sample, show_input = True, max_depth = 8))
    print(pytorch_model_summary.summary(model, sample, show_hierarchical = True, max_depth = 8))
    print(model)
    sample = torch.rand([8, 5, default_configs.gram_channel_num, 60, default_configs.slice_length])
    model.train()
    tmp = model(sample)
    for t in tmp:
        print(t.shape)
    del tmp
