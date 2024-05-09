import pytorch_model_summary
import torch

from configs import default_configs
from models.mine import Network, NetworkConfiguration

if __name__ == '__main__':
    
    model = Network(NetworkConfiguration())
    sample = torch.rand([8, 37, default_configs.gram_channel_num, 60, default_configs.slice_length])
    print(pytorch_model_summary.summary(model, sample, show_input = True, max_depth = 8))
    print(pytorch_model_summary.summary(model, sample, show_hierarchical = True, max_depth = 8))
    print(model)
    sample = torch.rand([8, 5, default_configs.gram_channel_num, 60, default_configs.slice_length])
    model.train()
    print('train_grl')
    tmp = model(sample, mode = 'train_grl', lambda0 = 0.1)
    for t in tmp:
        print(t.shape)
    del tmp
    for mode in ['train_ad_domain', 'whole', 'base', 'domain']:
        print(mode)
        tmp = model(sample, mode = 'whole')
        for t in tmp:
            print(t.shape)
        del tmp
