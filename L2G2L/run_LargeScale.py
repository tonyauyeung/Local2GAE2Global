"""
-*- coding = utf-8 -*-
@time:2022-05-06 14:58
@Author:Tony.SeoiHong.AuYeung
@File:run_LargeScale.py
@Software:PyCharm
"""
import argparse

import torch
import torch_geometric as tg
import os
import torch
import settings
from utils import load_data, train_test_split_Reconstruction, create_overlap_patches
import numpy as np
from train_scalable import train_l2gGAE, train_with_patch
import gc
import pickle
import time


def partition(dataset, is_scalable, repartition=False, num_patches='log',  **kwargs):
    if not os.path.exists('data/' + dataset + '/patch' + '/') or dataset == 'SBM' or repartition:
        data = load_data(dataset, **kwargs)
        train_data, val_data, test_data = train_test_split_Reconstruction(data)
        if num_patches == 'log':
            settings.num_patches = int(np.ceil(np.log(test_data.num_nodes)))
        elif num_patches == 'sqrt':
            settings.num_patches = int(np.ceil(np.sqrt(test_data.num_nodes)))
        elif num_patches == 'optimal':
            settings.num_patches = int(np.ceil(2 * np.sqrt((test_data.num_nodes * settings.latent_size) ** 3)))
        else:
            settings.num_patches = int(num_patches)
        patch_data, patch_graph = create_overlap_patches(train_data, device='cpu' if is_scalable else settings.device, num_patches=settings.num_patches)
        if dataset != 'SBM':
            try:
                os.makedirs(dataset + '/patch')
            except:
                None
            os.chdir(dataset + '/patch')
            with open('PatchData.pkl', 'wb') as f:
                dic = {'train_data': train_data, 'val_data':val_data, 'test_data': test_data, 'patch_data': patch_data, 'patch_graph': patch_graph}
                pickle.dump(dic, f)
            del dic
            os.chdir('..')
            os.chdir('..')
        os.chdir('..')
    else:
        with open('data/' + dataset + '/patch/PatchData.pkl', 'rb') as f:
           dic = pickle.load(f)
        train_data, val_data, test_data = dic['train_data'], dic['val_data'], dic['test_data']
        patch_data, patch_graph = dic['patch_data'], dic['patch_graph']
        if is_scalable:
            for p in patch_data:
                p.to('cpu')
        del dic
    del train_data
    gc.collect()

    return patch_data, patch_graph, val_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='L2G-FGAE for large scale dataset')
    parser.add_argument("--dataset", default='cora', type=str, help='Please specify a dataset')
    parser.add_argument("--device", default='cpu', type=str, help='Please specify your computing device')
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--model", default='all', type=str, help='all: run all models\n gae: run Loc2Glob-GAE\n'
                                                               'fgae: run Loc2Glob-FastGAE\n'
                                                               'vfgae: run Loc2Glob-VariationalFastGAE')
    parser.add_argument("--num_patches", default='2', type=str)
    parser.add_argument("--block_size", default=100, type=int)
    parser.add_argument("--block_num", default=10, type=int)
    parser.add_argument("--internal_prob", default=0.02, type=float)
    parser.add_argument("--external_prob", default=0.0001, type=float)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--res_state")
    args = parser.parse_args()
    dataset = args.dataset

    kwarg = {'block_size': args.block_size, 'block_num': args.block_num, 'internal_prob': args.internal_prob, 'external_prob': args.external_prob}

    settings.device = torch.device(args.device)
    settings.lr = args.lr
    settings.epoch = args.epoch

    flag = False if dataset in ['cora', 'cora_ml', 'Reddit2', 'Yelp', 'SBM'] else True
    patch_data, patch_graph, val_data, test_data = partition(dataset, is_scalable=flag, repartition=True,
                                                             num_patches=args.num_patches, **kwarg)

    print('Patch size: {}'.format(settings.num_patches))

    num_seeds = 10 if args.seed is None else 1
    os.chdir('results')
    if not os.path.exists(dataset + '{}/'.format(args.res_state)):
        os.makedirs(dataset + args.res_state)
    os.chdir(dataset + '{}/'.format(args.res_state))
    seeds = np.random.randint(0, 2022, num_seeds)
    f1, f2, f3 = [], [], []
    for i, seed in enumerate(seeds):
        foldername = 'seed_' + str(i) + '/' if num_seeds == 10 else 'seed_' + str(args.seed) + '/'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        os.chdir(foldername)
        tg.seed.seed_everything(seed)
        if args.model == 'all':
            print('Loc2Glob-GAE')
            t1 = time.time()
            f1.append(train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag))
            print('Loc2Glob-FastGAE')
            t2 = time.time()
            f2.append(train_with_patch('gae', patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag))
            print('Loc2Glob-Variational FastGAE')
            t3 = time.time()
            print(t2 - t1, t3 - t2)
            f3.append(train_with_patch('vgae', patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag))
            # print('L2G FGAE with subsampling')
            # train_with_patch('l2gfgae_single', patch_data, patch_graph, val_data, test_data, verbose=False,
            #                  is_subsample=True)
        elif args.model == 'gae':
            train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag)
        elif args.model == 'fgae':
            train_with_patch('gae', patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag)
        elif args.model == 'vfgae':
            train_with_patch('vgae', patch_data, patch_graph, val_data, test_data, verbose=False, is_scalable=flag)
        os.chdir('..')
    # if args.model == 'all':
    #     with open('glance.txt', 'w') as f:
    #         f.write('Loc2Glob-GAE\nROC: {}\tAP: {}\n\n'.format(np.mean([dic['roc_score'] for dic in f]), np.mean(f1['ap_score'])))
    #         f.write('Loc2Glob-FastGAE\nROC: {}\tAP: {}\n\n'.format(f2['roc_score'], f2['ap_score']))
    #         f.write('Loc2Glob-Variational FastGAE\nROC: {}\tAP: {}\n\n'.format(f3['roc_score'], f3['ap_score']))
    os.chdir('..')
    os.chdir('..')
