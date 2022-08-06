"""
-*- coding = utf-8 -*-
@time:2022-04-11 19:24
@Author:Tony.SeoiHong.AuYeung
@File:settings.py
@Software:PyCharm
"""
import torch_geometric as tg
import torch
tg.seed.seed_everything(2022)

epoch = 200
lr = 0.001
dropout = 0.
batch_size = None
hidden_size = 32
latent_size = 16
num_patches = 10
# min_overlap = latent_size + 1
min_overlap = 34
target_overlap = min_overlap * 2
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cpu')
skip = 10  # update the synchronisation matrices every "skip" steps
