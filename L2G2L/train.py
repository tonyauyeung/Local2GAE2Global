"""
-*- coding = utf-8 -*-
@time:2022-04-11 16:51
@Author:Tony.SeoiHong.AuYeung
@File:train.py
@Software:PyCharm
"""
import time
import pickle
import numpy as np
from typing import List
import torch
from torch import optim
import torch_geometric as tg
from torch_geometric.nn.models import GAE, InnerProductDecoder
from utils import load_data, train_test_split_Reconstruction, create_overlap_patches, synchronisation
from model import GCNEncoder, FastGAE, Loc2GlobFGAE
import settings

import local2global as l2g
from local2global_embedding.network import TGraph


def train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=True):
    num_features = test_data.num_features
    models = []
    optimizers = []
    train_loss = []
    valid_auc = []
    valid_ap = []
    times = []
    for i in range(len(patch_data)):
        encoder = GCNEncoder(num_features, settings.hidden_size, settings.latent_size)
        encoder.to(settings.device)
        decoder = InnerProductDecoder()
        models.append(GAE(encoder, decoder))
        optimizers.append(optim.Adam(encoder.parameters(), lr=settings.lr))

    if verbose:
        for e in range(settings.epoch):
            t1 = time.time()
            embeddings = []
            patch_list = []
            cur_loss = []
            for i, p in enumerate(patch_data):
                models[i].train()
                optimizers[i].zero_grad()
                z = models[i].encoder(p)
                # embeddings.append(z.data.numpy())
                embeddings.append(z)
                loss = models[i].recon_loss(z, p.edge_index)
                patch_list.append(l2g.utils.Patch(p.nodes.cpu().data.numpy(), z.cpu().data.numpy()))
                loss.backward()
                optimizers[i].step()
                cur_loss.append(loss.item())
                models[i].eval()
            t2 = time.time()
            with torch.no_grad():
                rotations, scales, translations, problem = synchronisation(patch_list, patch_graph, max_scale=None)
                global_emb = torch.zeros((test_data.num_nodes, settings.latent_size), dtype=torch.float64)
                for i in range(len(patch_data)):
                    embeddings[i] = torch.matmul(embeddings[i], rotations[i])
                    embeddings[i] = embeddings[i] / scales[i]
                    embeddings[i] += translations[i]
                for node, patch_list in enumerate(problem.patch_index):
                    global_emb[node] = torch.mean(
                        torch.stack([embeddings[p][problem.patches[p].index[node]] for p in patch_list]),
                        dim=0)
                roc_curr, ap_curr = models[0].test(global_emb, val_data.edge_index, val_data.negative_edge)

                print("Epoch:", '%04d' % (e + 1), "train_loss=", "{:.5f}".format(np.mean(cur_loss)),
                      "val_ap=", "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(t2 - t1))
                train_loss.append(np.mean(cur_loss))
                valid_ap.append(ap_curr)
                valid_auc.append(roc_curr)
                times.append(t2 - t1)

    else:
        embeddings = []
        patch_list = []
        for i, p in enumerate(patch_data):
            time_tmp = []
            loss_tmp = []
            for e in range(settings.epoch):
                t1 = time.time()
                models[i].train()
                models[i].zero_grad()
                z = models[i].encoder(p)
                loss = models[i].recon_loss(z, p.edge_index)
                loss.backward()
                cur_loss = loss.item()
                optimizers[i].step()
                loss_tmp.append(cur_loss)
                time_tmp.append(time.time() - t1)
            print('Patch {} is well trained. Time/epoch: {}, Loss: {}'.
                  format(i + 1, np.mean(time_tmp), loss_tmp[-1]))
            train_loss.append(loss_tmp)
            times.append(time_tmp)
            z = models[i].encoder(p)
            embeddings.append(z)
            patch_list.append(l2g.utils.Patch(p.nodes.cpu().data.numpy(), z.cpu().data.numpy()))

    t1 = time.time()
    patch_list = []
    for i, p in enumerate(patch_data):
        patch_list.append(l2g.utils.Patch(p.nodes.cpu().data.numpy(), models[i].encoder(p).cpu().data.numpy()))
    rotations, scales, translations, problem = synchronisation(patch_list, patch_graph, max_scale=None)
    global_emb = torch.zeros((test_data.num_nodes, settings.latent_size), dtype=torch.float64)
    for i in range(len(patch_data)):
        embeddings[i] = torch.matmul(embeddings[i], rotations[i])
        embeddings[i] = embeddings[i] / scales[i]
        embeddings[i] += translations[i]
    for node, patch_list in enumerate(problem.patch_index):
        global_emb[node] = torch.mean(
            torch.stack([embeddings[p][problem.patches[p].index[node]] for p in patch_list]),
            dim=0)
    t2 = time.time()
    roc_score, ap_score = models[0].test(global_emb, test_data.edge_index, test_data.negative_edge)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    dic = {'train_loss': train_loss, 'times': times, 'valid_ap': valid_ap, 'valid_auc': valid_auc,
           'roc_score': roc_score, 'ap_score': ap_score, 'sync_time': t2 - t1}
    if verbose:
        with open('l2gGAE_verbose.pkl', 'wb') as f:
            pickle.dump(dic, f)
    else:
        with open('l2gGAE_efficiency.pkl', 'wb') as f:
            pickle.dump(dic, f)


def train_with_full(mode: str, train_data, val_data, test_data, verbose=True):
    train_loss = []
    valid_auc = []
    valid_ap = []
    times = []
    num_features = train_data.num_features
    encoder = GCNEncoder(num_features, settings.hidden_size, settings.latent_size)
    encoder.to(settings.device)
    decoder = InnerProductDecoder()
    if mode.lower() == 'gae':
        model = GAE(encoder, decoder)
    elif mode.lower() == 'fgae':
        model = FastGAE(encoder, decoder, sample_style='degree')
    else:
        raise ValueError("NO SUCH MODE!\nSELECT 'gae' OR 'fgae' PLEASE")
    optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    for e in range(settings.epoch):
        t1 = time.time()
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data)
        if mode.lower() == 'gae':
            loss = model.recon_loss(z, train_data.edge_index)
        else:
            sub_nodes, sub_edge_index, _ = model.subgraph_sampler(train_data, sample_size=None)
            loss = model.recon_loss(z[sub_nodes], sub_edge_index)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        cur_loss = loss.item()
        model.eval()
        times.append(t2 - t1)
        if verbose:
            with torch.no_grad():
                roc_curr, ap_curr = model.test(z, val_data.edge_index, val_data.neg_edge_index)
            print("Epoch:", '%04d' % (e + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(t2 - t1))
            train_loss.append(cur_loss)
            valid_ap.append(ap_curr)
            valid_auc.append(roc_curr)
    roc_score, ap_score = model.test(z, test_data.edge_index, test_data.neg_edge_index)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    dic = {'train_loss': train_loss, 'times': times, 'valid_ap': valid_ap, 'valid_auc': valid_auc,
           'roc_score': roc_score, 'ap_score': ap_score}
    with open(mode.lower() + '.pkl', 'wb') as f:
        pickle.dump(dic, f)
    return dic


def train_with_patch(mode: str, patch_data: List[tg.data.data.Data], patch_graph: TGraph,
                     val_data, test_data, verbose=True):
    train_loss = []
    valid_auc = []
    valid_ap = []
    times = []
    num_features = test_data.num_features
    encoder = GCNEncoder(num_features, settings.hidden_size, settings.latent_size)
    encoder.to(settings.device)
    encoder.device = settings.device
    decoder = InnerProductDecoder()
    patch_size = len(patch_data)
    if mode.lower() == 'l2gfgae_single':
        model = Loc2GlobFGAE(encoder, decoder, mode='single')
        optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    elif mode.lower() == 'l2gfgae_multi':
        model = Loc2GlobFGAE(encoder, decoder, mode='multi')
        optimizers = [optim.Adam(model.encoders[i].parameters()) for i in range(patch_size)]

    for e in range(settings.epoch):
        t1 = time.time()
        model.train()
        if mode.lower() == 'l2gfgae_single':
            optimizer.zero_grad()
        else:
            for optimizer in optimizers:
                optimizer.zero_grad()
        embeddings = model.encode(patch_data, patch_graph)
        loss = model.recon_loss_patch(embeddings, patch_data)
        loss.backward()
        # optimizer.step()
        if mode.lower() == 'l2gfgae_single':
            optimizer.step()
        else:
            for optimizer in optimizers:
                optimizer.step()
        cur_loss = loss.item()
        encoder.eval()
        t2 = time.time()
        with torch.no_grad():
            global_emb = model.global_embedding(embeddings, test_data.num_nodes)
            if verbose:
                roc_curr, ap_curr = model.test(global_emb, val_data.edge_index, val_data.negative_edge)

                print("Epoch:", '%04d' % (e + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(t2 - t1))
                train_loss.append(cur_loss)
                times.append(t2 - t1)
                valid_ap.append(ap_curr)
                valid_auc.append(roc_curr)

    roc_score, ap_score = model.test(global_emb, test_data.edge_index, test_data.negative_edge)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    dic = {'train_loss': train_loss, 'times': times, 'valid_ap': valid_ap, 'valid_auc': valid_auc,
           'roc_score': roc_score, 'ap_score': ap_score}
    with open(mode.lower() + '.pkl', 'wb') as f:
        pickle.dump(dic, f)


if __name__ == '__main__':
    data = load_data('cora_ML')
    train_data, val_data, test_data = train_test_split_Reconstruction(data)
    patch_data, patch_graph = create_overlap_patches(train_data)
    print('GAE:')
    # train_with_full('gae', train_data, val_data, test_data, verbose=False)
    # print('FastGAE:')
    # train_with_full('fgae', train_data, val_data, test_data, verbose=False)
    # print('Loc2Glob GAE')
    # TODO: Solve the unstability of the synchronisation algorithm
    train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=False)
    # print('Loc2Glob FastGAE in single model mode')
    # train_with_patch('l2gfgae_single', patch_data, patch_graph, val_data, test_data, verbose=False)
    # print('Loc2Glob FastGAE in multi model mode')
    # train_with_patch('l2gfgae_multi', patch_data, patch_graph, val_data, test_data, verbose=False)
