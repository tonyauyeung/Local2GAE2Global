"""
-*- coding = utf-8 -*-
@time:2022-04-11 20:33
@Author:Tony.SeoiHong.AuYeung
@File:functional.py
@Software:PyCharm
"""
import torch
import torch.nn.functional as F


def loss_GAE(input, target, norm=None, pos_weight=torch.tensor(1)):
    if norm is None:
        norm = len(input)
    cost = norm * F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight)
    return cost


def loss_VGAE(input, target, mu, logvar, n_nodes, norm=None, pos_weight=torch.tensor(1)):
    if norm is None:
        norm = len(input)
    cost = loss_GAE(input, target, norm, pos_weight=pos_weight)
    KL = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL