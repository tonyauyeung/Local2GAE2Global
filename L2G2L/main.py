"""
-*- coding = utf-8 -*-
@time:2022-04-12 17:48
@Author:Tony.SeoiHong.AuYeung
@File:main.py
@Software:PyCharm
"""
from utils import *
import argparse
from train import *
import os
import torch
import pickle
import torch_geometric as tg


# # TODO: Add help statementsR
# parser = argparse.ArgumentParser(description='Loc2Glob project')
# parser.add_argument('--epoch', default=200, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--dropout', default=0., type=float)
# parser.add_argument('--batch_size', default=None)
# parser.add_argument('--hidden_size', default=32, type=int)
# parser.add_argument('--latent_size', default=16, type=int)
# parser.add_argument('--num_patches', default=10, type=int)
# parser.add_argument('--min_overlap', default=34, type=int)
# parser.add_argument('--target_overlap', default=68, type=int)
# parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def main(scale='medium', verbose=False):
    if scale == 'small':
        # datasets = ['cora', 'cora_ML', 'Amazon_computers', 'Amazon_photos']  # small scale
        datasets = ['cora_ML']
    elif scale == 'medium':
        datasets = ['Reddit', 'Reddit2']  # medium scale
    else:
        datasets = ['Yelp', 'AmazonProduct']  # large scale
    seeds = np.random.randint(0, 2022, 10)
    for dataset in datasets:
        data = load_data(dataset)
        data.to(settings.device)
        os.chdir('..')
        if not os.path.exists('results/'):
            os.makedirs('results')
        os.chdir('results')
        if not os.path.exists(dataset + '/'):
            os.makedirs(dataset)
        os.chdir(dataset + '/')
        train_data, val_data, test_data = train_test_split_Reconstruction(data)
        patch_data, patch_graph = create_overlap_patches(train_data)
        for i, seed in enumerate(seeds):
            if not os.path.exists('seed_' + str(i) + '/'):
                os.makedirs('seed_' + str(i) + '/')
            os.chdir('seed_' + str(i) + '/')
            tg.seed.seed_everything(seed)
            # train_with_full('gae', train_data, val_data, test_data, verbose=verbose)
            # train_with_full('fgae', train_data, val_data, test_data, verbose=verbose)
            # train_l2gGAE(patch_data, patch_graph, val_data, test_data, verbose=verbose)
            train_with_patch('l2gfgae_single', patch_data, patch_graph, val_data, test_data, verbose=verbose)
            # train_with_patch('l2gfgae_multi', patch_data, patch_graph, val_data, test_data, verbose=verbose)
            os.chdir('..')
        os.chdir('..')
        os.chdir('..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='SBM', type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--block_size", default=100, type=int)
    parser.add_argument("--block_num", default=10, type=int)
    parser.add_argument("--statement", default='', type=str)
    parser.add_argument("--internal_prob", default=0.02, type=float)
    parser.add_argument("--external_prob", default=0.0001, type=float)
    seeds = np.random.randint(0, 2022, 10)
    tmps = []
    args = parser.parse_args()
    if args.dataset == 'small':
        datasets = ['cora_ml', 'cora', 'SBM']
        kwargs = {'block_size': 100, 'block_num': 100}
    else:
        datasets = [args.dataset]
        kwarg = {'block_size': args.block_size, 'block_num': args.block_num, 'internal_prob': args.internal_prob,
                 'external_prob': args.external_prob}
    for dataset in datasets:
        tmp = []
        for seed in seeds:
            # for dataset in ['cora_ml', 'cora', 'Reddit2', 'Yelp']:
            tg.seed_everything(seed)
            print(dataset)
            data = load_data(dataset, **kwargs)
            data.to(settings.device)
            train_data, val_data, test_data = train_test_split_Reconstruction(data)
            tmp.append(train_with_full('fgae', train_data, val_data, test_data, verbose=False))
            os.chdir('..')
        tmps.append(tmp)
    with open('results/' + args.statement + 'glance_fgae.txt', 'w') as f:
        for i, dataset in enumerate(datasets):
            tmp1 = [tmps[i][j]['roc_score'] for j in range(10)]
            tmp2 = [tmps[i][j]['ap_score'] for j in range(10)]
            tmp3 = [np.mean(tmps[i][j]['times']) for j in range(10)]
            roc = np.mean(tmp1)
            ap = np.mean(tmp2)
            times = np.mean(tmp3)
            f.writelines("{}\n\tROC: {:.5f}±{:.5f}\tAP: {:.5f}±{:.5f}\ttime: {:.5f}±{:.5f}/epoch\n\n".format(dataset, roc, np.std(tmp1), ap, np.std(tmp2), times, np.std(tmp3)))