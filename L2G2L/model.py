"""
-*- coding = utf-8 -*-
@time:2022-04-11 19:10
@Author:Tony.SeoiHong.AuYeung
@File:model.py
@Software:PyCharm
"""
import numpy as np
from copy import deepcopy
from typing import List, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.nn.models import GAE, VGAE
import local2global as l2g
from local2global_embedding.network import TGraph

# TODO: Add dropout layer
import utils
import settings

T = TypeVar('T', bound='Module')


class GCNEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.):
        super(GCNEncoder, self).__init__()
        self.gc1 = GCNConv(feature_size, hidden_size)
        self.gc2 = GCNConv(hidden_size, latent_size)

    def forward(self, data):
        output = F.relu(self.gc1(data.x, data.edge_index, data.edge_weight))
        return self.gc2(output, data.edge_index, data.edge_weight)


class VGAE_Encoder(GCNEncoder):
    def __init__(self, feature_size, hidden_size, latent_size, dropout=0.):
        super(VGAE_Encoder, self).__init__(feature_size, hidden_size, latent_size, dropout)
        self.gc3 = GCNConv(hidden_size, latent_size)

    def forward(self, data):
        output = F.relu(self.gc1(data.x, data.edge_index, data.edge_weight))
        mu = self.gc2(output, data.edge_index, data.edge_weight)
        logstd = self.gc3(output, data.edge_index, data.edge_weight)
        return mu, logstd


class FastGAE(GAE):
    def __init__(self, encoder, decoder=None, sample_style='uniform'):
        super(FastGAE, self).__init__(encoder, decoder)
        self.sample_style = sample_style
        if sample_style == 'uniform':
            self.sampler = lambda x: np.random.choice(**x, replace=False)
        else:
            self.sampler = lambda x: np.random.choice(**x, replace=False)

    def subgraph_sampler(self, data, sample_size=None, alpha=0.1, gamma=1, epsilon=1e-3):
        if sample_size is None:
            sample_size = int(np.sqrt(data.num_nodes * (-np.log(alpha / 2) * np.log(epsilon) ** 2)
                                      / (2 * gamma ** 2)))
        # TODO: Implement for another sampler
        if self.sample_style == 'uniform':
            p = np.ones(data.num_nodes, dtype=np.float) / data.num_nodes
        elif self.sample_style == 'degree':
            p = tg.utils.degree(data.edge_index.flatten(), data.num_nodes).cpu().data.numpy()
            p /= np.sum(p)
        else:
            p = np.ones_like(data.num_nodes, dtype=np.float) / data.num_nodes
        return utils.subgraph_sampler(data, sample_size, p)


class Loc2GlobFGAE(GAE):
    def __init__(self, encoder, decoder=None, mode='single', patch_size=10, is_scalable=False):
        """
        Local2Global FastGAE model with two different modes. And the decoder must be a non-parametric function,
        which means that we're not going to update decoder during training,
        e.g. the default one - InnerProductDecoder()
        """
        decoder = InnerProductDecoder() if decoder is None else decoder
        super(Loc2GlobFGAE, self).__init__(encoder, decoder)
        self.problem = None
        self.mode = mode
        if mode != 'single':
            self.patch_size = patch_size
            self.encoders = [deepcopy(encoder)] * patch_size
            self.decoders = [deepcopy(decoder)] * patch_size
            if not hasattr(encoder, 'device'):
                encoder.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for e in self.encoders:
                e.to(encoder.device)
        self.device = encoder.device
        self.is_scalable = is_scalable
        self.rotations = None
        self.scales = None
        self.translations = None

    def encode(self, patch_data: List[tg.data.data.Data], patch_graph: TGraph, is_synchronize=True, **kwargs) \
            -> List[torch.Tensor]:
        embeddings = []
        patch_list = []
        for i, p in enumerate(patch_data):
            if self.is_scalable:
                p.to(self.device)

            if self.mode == 'single':
                z = self.encoder(p, **kwargs)
            else:
                z = self.encoders[i](p, **kwargs)
            embeddings.append(z)
            if is_synchronize:
                patch_list.append(l2g.utils.Patch(p.nodes.cpu().data.numpy(), z.cpu().data.numpy()))
            if self.is_scalable:
                p.to('cpu')
        if is_synchronize:
            self.rotations, self.scales, self.translations, self.problem = \
                utils.synchronisation(patch_list, patch_graph, max_scale=None, is_translate=False)
        for i in range(len(patch_data)):
            embeddings[i] = torch.matmul(embeddings[i], self.rotations[i])
            embeddings[i] = embeddings[i] / self.scales[i]
            embeddings[i] += self.translations[i]
        return embeddings

    def recon_loss_patch(self, embeddings: List[torch.Tensor], patch_data: List[tg.data.data.Data]):
        loss = torch.tensor(0, dtype=torch.float64, device=self.device)
        s = 0
        for i, p in enumerate(patch_data):
            if self.is_scalable:
                p.to(self.device)
            if hasattr(p, 'neg_edge_index'):
                loss += self.recon_loss(embeddings[i], p.edge_index, p.neg_edge_index) * p.num_nodes
            else:
                loss += self.recon_loss(embeddings[i], p.edge_index) * p.num_nodes
            s += p.num_nodes
            if self.is_scalable:
                p.to('cpu')
        loss /= s
        return loss

    @staticmethod
    def global_embedding(problem, embeddings: List[torch.Tensor], num_nodes: int):
        global_emb = torch.zeros((num_nodes, settings.latent_size), dtype=torch.float64)
        for node, patch_list in enumerate(problem.patch_index):
            global_emb[node] = torch.mean(
                torch.stack([embeddings[p][problem.patches[p].index[node]] for p in patch_list]), dim=0)
        return global_emb

    def recon_loss_patch_fast(self, embeddings: List[torch.Tensor], patch_data: List[tg.data.data.Data], probs=None):
        loss = torch.tensor(0, dtype=torch.float64, device=self.device)
        s = 0
        for i, p in enumerate(patch_data):
            if self.is_scalable:
                p.to(self.device)
            sample_size = int(np.sqrt(p.num_nodes))
            # sample_size = int(p.num_nodes * 0.5)
            sub_nodes, sub_edge_index, _ = utils.subgraph_sampler(p, sample_size, probs[i])
            if sub_edge_index.size()[1] > 0:
                loss += self.recon_loss(embeddings[i][sub_nodes.long()], sub_edge_index.long()) * p.num_nodes
            s += p.num_nodes
            if self.is_scalable:
                p.to('cpu')
        loss /= s
        return loss


class Loc2GlobVarFGAE(VGAE):
    def __init__(self, encoder, decoder=None, patch_size=10, is_scalable=False):
        """
        Local2Global Variational FastGAE model.
        """
        decoder = InnerProductDecoder() if decoder is None else decoder
        super(Loc2GlobVarFGAE, self).__init__(encoder, decoder)
        self.problem = None
        self.device = encoder.device
        self.is_scalable = is_scalable
        self.rotations = None
        self.scales = None
        self.translations = None
        self.__mus__ = [None] * patch_size
        self.__logstds__ = [None] * patch_size

    def encode(self, patch_data: List[tg.data.data.Data], patch_graph: TGraph, is_synchronize=True, **kwargs):
        embeddings = []
        patch_list = []
        for i, p in enumerate(patch_data):
            if self.is_scalable:
                p.to(self.device)
            self.__mus__[i], self.__logstds__[i] = self.encoder(p, **kwargs)
            embeddings.append(self.reparametrize(self.__mus__[i], self.__logstds__[i]))
            if is_synchronize:
                patch_list.append(l2g.utils.Patch(p.nodes.cpu().data.numpy(), embeddings[-1].cpu().data.numpy()))
            if self.is_scalable:
                p.to('cpu')

        if is_synchronize:
            self.rotations, self.scales, self.translations, self.problem = \
                utils.synchronisation(patch_list, patch_graph, max_scale=None)
        for i in range(len(patch_data)):
            embeddings[i] = torch.matmul(embeddings[i], self.rotations[i])
            embeddings[i] += self.translations[i]
            # self.__mus__[i] = torch.matmul(self.__mus__[i], self.rotations[i])
            # # self.__mus__[i] = self.__mus__[i] / self.scales[i]
            # self.__mus__[i] += self.translations[i]
            # embeddings.append(self.reparametrize(self.__mus__[i], self.__logstds__[i]))
        return embeddings

    def recon_loss_patch(self, embeddings: List[torch.Tensor], patch_data: List[tg.data.data.Data]):
        loss = torch.tensor(0, dtype=torch.float64, device=self.device)
        s = 0
        for i, p in enumerate(patch_data):
            if self.is_scalable:
                p.to(self.device)
            kl = self.kl_loss(self.__mus__[i], self.__logstds__[i])
            tmp = p.neg_edge_index if hasattr(p, 'neg_edge_index') else None
            loss += (self.recon_loss(embeddings[i], p.edge_index, tmp) + kl) * p.num_nodes
            s += p.num_nodes
            if self.is_scalable:
                p.to('cpu')
        loss /= s
        return loss

    @staticmethod
    def global_embedding(problem, embeddings: List[torch.Tensor], num_nodes: int):
        return Loc2GlobFGAE.global_embedding(problem, embeddings, num_nodes)