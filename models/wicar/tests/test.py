import pytorch_model_summary
import torch

from configs import default_configs
from models.wicar import Network, NetworkConfiguration


if __name__ == '__main__':
    model = Network(NetworkConfiguration())
    sample = torch.rand([16, default_configs.gram_channel_num, 61, 400])
    print(pytorch_model_summary.summary(model, sample, show_input = True, max_depth = 8))
    print(pytorch_model_summary.summary(model, sample, show_hierarchical = True, max_depth = 8))
    print(model)
    sample = torch.rand([16, default_configs.gram_channel_num, 61, 400])
    model.train()
    print('train_grl')
    tmp = model(sample, mode = 'train_grl', lambda0 = 0.1)
    for t in tmp:
        if type(t) == tuple:
            print(t[0].shape, t[1].shape)
        else:
            print(t.shape)
    del tmp
    for mode in ['whole', 'base', 'domain']:
        print(mode)
        tmp = model(sample, mode = 'whole')
        for t in tmp:
            if type(t) == tuple:
                print(t[0].shape, t[1].shape)
            else:
                print(t.shape)
        del tmp
