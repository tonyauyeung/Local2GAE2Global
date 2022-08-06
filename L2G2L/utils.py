"""
-*- coding = utf-8 -*-
@time:2022-04-11 15:52
@Author:Tony.SeoiHong.AuYeung
@File:utils.py
@Software:PyCharm
"""
import torch
import torch_geometric as tg
from torch_geometric.utils import negative_sampling, add_self_loops, sort_edge_index
from torch_geometric.transforms import RandomLinkSplit
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

import local2global as l2g

from local2global_embedding.patches import create_patch_data
from local2global_embedding.clustering import fennel_clustering
from local2global_embedding.network import TGraph

import settings


def load_data(dataset='cora', **kwargs):
    # TODO: Implement interface for more datasets.
    if not os.path.exists('data/'):
        os.makedirs('data')

    os.chdir('data/')
    if dataset == 'cora':
        data = tg.datasets.CitationFull(root='', name='cora').data
    elif dataset == 'cora_ml':
        data = tg.datasets.CitationFull(root='', name='Cora_ML').data
    elif dataset == 'Amazon_computers':
        data = tg.datasets.Amazon(root='Amazon/', name='Computers').data
    elif dataset == 'Amazon_photos':
        data = tg.datasets.Amazon(root='Amazon/', name='Photo').data
    elif dataset == 'Reddit':
        data = tg.datasets.Reddit(root='Reddit/').data
    elif dataset == 'Reddit2':
        data = tg.datasets.Reddit2(root='Reddit2/').data
    elif dataset == 'Yelp':
        data = tg.datasets.Yelp(root='Yelp/').data
    elif dataset == 'AmazonProduct':
        data = tg.datasets.AmazonProduct(root='AmazonProduct/').data
    elif dataset == 'SBM':
        data = synthetic_data(**kwargs)
    else:
        raise ValueError("This dataset doesn't exist or is unavailable now.")
    return data


def synthetic_data(block_size=1000, block_num=100, internal_prob=0.02, external_prob=0.0001):
    g = nx.stochastic_block_model([block_size, ] * block_num,
                                  np.eye(block_num) * internal_prob + (1-np.eye(block_num)) * external_prob)
    edge_index = torch.LongTensor(np.array(g.edges()).T)
    # edge_index = tg.utils.stochastic_blockmodel_graph([block_size, ] * block_num, np.eye(block_num) * internal_prob +
    #                                                   (1 - np.eye(block_num)) * external_prob)
    num_nodes = block_size * block_num
    tmp = torch.eye(block_num)
    x = torch.concat([tmp[i].repeat(block_size, 1) for i in range(block_num)])
    return tg.data.Data(edge_index=edge_index, num_nodes=num_nodes, x=x)


def get_roc_score(embedding, edge_index, negative_edge_index):
    # Predict on test set of edges
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    pred_pos = []
    pred_neg = []
    for e in edge_index:
        pred_pos.append(sigmoid(np.dot(embedding[e[0]], embedding[e[1]])))
    for e in negative_edge_index:
        pred_neg.append(sigmoid(np.dot(embedding[e[0]], embedding[e[1]])))
    pred = pred_pos + pred_neg
    label = np.concatenate((np.ones_like(pred_pos), np.zeros_like(pred_neg)))
    roc_score = roc_auc_score(label, pred)
    ap_score = average_precision_score(label, pred)

    return roc_score, ap_score


def train_test_split_Reconstruction(data: tg.data.data.Data, num_val=0.05, num_test=0.1, is_undirected=True):
    transform = RandomLinkSplit(num_val=num_val, num_test=num_test, is_undirected=is_undirected)
    train_data, val_data, test_data = transform(data)
    num_nodes = data.num_nodes
    # train_data.negative_edge = negative_sampling(train_data.edge_index, num_nodes=num_nodes)
    # val_data.edge_index = edge_diff(val_data.edge_index, train_data.edge_index, num_nodes)
    test_data.edge_index = edge_diff(test_data.edge_index, train_data.edge_index, num_nodes)
    val_data.edge_index, _ = add_self_loops(val_data.edge_index)
    test_data.edge_index, _ = add_self_loops(test_data.edge_index)
    train_data.neg_edge_index = negative_sampling(train_data.edge_index, num_nodes=num_nodes,
                                                  num_neg_samples=val_data.edge_index.shape[1])
    val_data.neg_edge_index = negative_sampling(data.edge_index, num_nodes=num_nodes,
                                                num_neg_samples=val_data.edge_index.shape[1])
    test_data.neg_edge_index = negative_sampling(data.edge_index, num_nodes=num_nodes,
                                                 num_neg_samples=test_data.edge_index.shape[1])
    return train_data, val_data, test_data


def create_overlap_patches(data: tg.data.data.Data, num_patches=10, min_overlap=settings.latent_size+1,
                           target_overlap=(settings.latent_size+1)*2, device=settings.device):
    if data.edge_index.device != torch.device('cpu'):
        data.to('cpu')
    graph = TGraph(data.edge_index)
    clusters = fennel_clustering(graph, num_clusters=num_patches)
    patch_data, patch_graph = create_patch_data(data=data, partition_tensor=clusters, min_overlap=min_overlap,
                                                target_overlap=target_overlap)

    # TODO: Make sure the all attributes of p is the same as that of data, not only "x" for the current version
    for p in patch_data:
        p.x = data.x[p.nodes]
        p.neg_edge_index = negative_sampling(p.edge_index, num_nodes=p.num_nodes,
                                             num_neg_samples=p.edge_index.shape[1])
        p.to(device)
    patch_graph.edge_index = patch_graph.edge_index.to(device)
    # if data.edge_index.device == torch.device('cpu'):
    #     data.to(device)
    return patch_data, patch_graph


def synchronisation(patch_list, patch_graph, max_scale=None, is_translate=True):
    # TODO: Process the synchronisation on GPU core...
    patch_edge = patch_graph.edge_index.cpu().data.numpy().T
    patch_edge = [tuple(e) for e in patch_edge]
    problem = l2g.utils.AlignmentProblem(patch_list, patch_edge)
    rotations = torch.tensor(problem.calc_synchronised_rotations(), device=settings.device).float()
    scales = torch.ones(len(patch_list), device=settings.device) if max_scale is None \
        else torch.tensor(problem.calc_synchronised_scales(max_scale)).float()
    if is_translate:
        translations = torch.tensor(problem.calc_synchronised_translations(), device=settings.device).float()
    else:
        translations = torch.zeros((len(patch_list), patch_list[0].coordinates.shape[1]), dtype=torch.float64, device=settings.device)
    return rotations, scales, translations, problem


def subgraph_sampler(data: tg.data.data.Data, sample_size, p=None):
    # TODO: Implemented the weighted sampling without replacement purely using torch.
    nodes = list(np.sort(np.random.choice(np.arange(data.num_nodes), sample_size, p=p, replace=False)))
    sub_edge_index, sub_edge_attr = tg.utils.subgraph(nodes, data.edge_index)
    sub_edge_index, sub_edge_attr, _ = tg.utils.remove_isolated_nodes(sub_edge_index, sub_edge_attr)
    nodes = torch.tensor(nodes, device=settings.device)
    return nodes.long(), sub_edge_index, sub_edge_attr


def edge_diff(edge_index1, edge_index2, num_nodes=None):
    """
    Computing the set difference: edge_index1 - edge_index2, which is used for validation and evaluation
    """
    edge_index1 = set(map(tuple, sort_edge_index(edge_index1, num_nodes=num_nodes).cpu().data.numpy().T))
    edge_index2 = set(map(tuple, sort_edge_index(edge_index2, num_nodes=num_nodes).cpu().data.numpy().T))
    diff = list(map(torch.LongTensor, edge_index1 - edge_index2))
    return torch.stack(diff).T.to(settings.device)


if __name__ == '__main__':
    data = load_data('Reddit')
    print(1)
    # train_data, val_data, test_data = train_test_split_Reconstruction(data)
    # patch_data, patch_graph = create_overlap_patches(train_data)