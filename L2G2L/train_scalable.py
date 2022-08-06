"""
-*- coding = utf-8 -*-
@time:2022-05-06 13:38
@Author:Tony.SeoiHong.AuYeung
@File:train_scalable.py
@Software:PyCharm
"""
import os
import time
import pickle
import numpy as np
from typing import List
import torch
from torch import optim
import torch_geometric as tg
from torch_geometric.nn.models import GAE, InnerProductDecoder

import utils
from utils import load_data, train_test_split_Reconstruction, create_overlap_patches, synchronisation
from model import GCNEncoder, FastGAE, Loc2GlobFGAE, Loc2GlobVarFGAE, VGAE_Encoder
import settings

import local2global as l2g
from local2global_embedding.network import TGraph

import tqdm

import gc


def train_with_patch(mode: str, patch_data: List[tg.data.data.Data], patch_graph: TGraph,
                     val_data, test_data, verbose=True, is_subsample=False, is_scalable=False):
    train_loss = []
    valid_auc = []
    valid_ap = []
    times = []
    num_features = test_data.num_features
    patch_size = len(patch_data)
    decoder = InnerProductDecoder()
    if mode == 'gae':
        encoder = GCNEncoder(num_features, settings.hidden_size, settings.latent_size)
        encoder.to(settings.device)
        encoder.device = settings.device
        model = Loc2GlobFGAE(encoder, decoder, mode='single', is_scalable=is_scalable, patch_size=patch_size)
    elif mode == 'vgae':
        encoder = VGAE_Encoder(num_features, settings.hidden_size, settings.latent_size)
        encoder.to(settings.device)
        encoder.device = settings.device
        model = Loc2GlobVarFGAE(encoder, decoder, patch_size=patch_size, is_scalable=is_scalable)
    else:
        raise ValueError("Invalid model type...")
    optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    if is_subsample:
        probs = []
        for p in patch_data:
            tmp = tg.utils.degree(p.edge_index.flatten(), p.num_nodes).cpu().data.numpy()
            tmp /= np.sum(tmp)
            probs.append(tmp)
    for e in tqdm.tqdm(range(settings.epoch)):
        t1 = time.time()
        model.train()
        optimizer.zero_grad()
        if e % settings.skip == 0:
            embeddings = model.encode(patch_data, patch_graph, is_synchronize=True)
        else:
            embeddings = model.encode(patch_data, patch_graph, is_synchronize=False)
        if is_subsample:
            loss = model.recon_loss_patch_fast(embeddings, patch_data, probs)
        else:
            loss = model.recon_loss_patch(embeddings, patch_data)
        loss.backward()
        optimizer.step()
        cur_loss = loss.item()
        encoder.eval()
        t2 = time.time()
        train_loss.append(cur_loss)
        times.append(t2 - t1)
        if verbose:
            with torch.no_grad():
                global_emb = model.global_embedding(model.problem, model.__mus__, test_data.num_nodes) \
                    if mode == 'vgae' else model.global_embedding(model.problem, embeddings, test_data.num_nodes)
                roc_curr, ap_curr = model.test(global_emb, val_data.edge_index, val_data.neg_edge_index)

                print("Epoch:", '%04d' % (e + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(t2 - t1))
                valid_ap.append(ap_curr)
                valid_auc.append(roc_curr)

    model.eval()
    global_emb = model.global_embedding(model.problem, model.__mus__, test_data.num_nodes) if mode == 'vgae' \
        else model.global_embedding(model.problem, embeddings, test_data.num_nodes)
    roc_score, ap_score = model.test(global_emb, test_data.edge_index, test_data.neg_edge_index)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    print('average training time/epoch: {}'.format(np.mean(times)))
    dic = {'train_loss': train_loss, 'times': times, 'valid_ap': valid_ap, 'valid_auc': valid_auc,
           'roc_score': roc_score, 'ap_score': ap_score}
    del model
    gc.collect()
    with open('Loc2Glob-fast-' + mode.lower() + '.pkl', 'wb') as f:
        pickle.dump(dic, f)
    return dic


def train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=True, is_scalable=False):
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
                roc_curr, ap_curr = models[0].test(global_emb, val_data.edge_index, val_data.neg_edge_index)

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
            if is_scalable:
                p.to(settings.device)
            time_tmp = []
            loss_tmp = []
            for e in range(settings.epoch):
                t1 = time.time()
                models[i].train()
                models[i].zero_grad()
                z = models[i].encoder(p)
                loss = models[i].recon_loss(z, p.edge_index, p.neg_edge_index)
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
            if is_scalable:
                p.to('cpu')
            patch_list.append(l2g.utils.Patch(p.nodes.data.cpu().numpy(), z.cpu().data.numpy()))

    t1 = time.time()
    patch_list = []
    for i, p in enumerate(patch_data):
        tmp = p.nodes.data.cpu().numpy()
        if is_scalable:
            p.to(settings.device)
        patch_list.append(l2g.utils.Patch(tmp, models[i].encoder(p).cpu().data.numpy()))
        if is_scalable:
            p.to('cpu')
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
    roc_score, ap_score = models[0].test(global_emb, test_data.edge_index, test_data.neg_edge_index)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    print('average training time/epoch: {}'.format(np.mean(times)))
    dic = {'train_loss': train_loss, 'times': times, 'valid_ap': valid_ap, 'valid_auc': valid_auc,
           'roc_score': roc_score, 'ap_score': ap_score, 'sync_time': t2 - t1}
    if verbose:
        with open('l2gGAE_verbose.pkl', 'wb') as f:
            pickle.dump(dic, f)
    else:
        with open('l2gGAE_efficiency.pkl', 'wb') as f:
            pickle.dump(dic, f)
    del models
    gc.collect()
    return dic


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
