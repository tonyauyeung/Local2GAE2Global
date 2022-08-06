import pickle
import argparse
import numpy as np


def glance(path, statement='', num_seeds=10):
    dics = []
    if args.files == 'all':
        files = ['l2gGAE_efficiency.pkl', 'Loc2Glob-fast-gae.pkl', 'Loc2Glob-fast-vgae.pkl']
        dics = {0: [], 1: [], 2: []}
    elif args.files == 'novfgae':
        files = ['l2gGAE_efficiency.pkl', 'Loc2Glob-fast-gae.pkl']
        dics = {0: [], 1:[]}
    for i in range(num_seeds):
        subpath = '/seed_{}'.format(i)
        for j, file in enumerate(files):
            with open(path + subpath + '/' + file, 'rb') as f:
                dic = pickle.load(f)
            dics[j].append(dic)
    roc_scores = []
    ap_scores = []
    time_epoch = []
    for j in range(len(files)):
        tmp1 = [dics[j][i]['roc_score'] for i in range(10)]
        tmp2 = [dics[j][i]['ap_score'] for i in range(10)]
        tmp3 = [np.mean(dics[j][i]['times']) for i in range(10)]
        roc_scores.append([np.mean(tmp1), np.std(tmp1)])
        ap_scores.append([np.mean(tmp2), np.std(tmp2)])
        time_epoch.append([np.mean(tmp3), np.std(tmp3)])

    with open(path + '/glance' + statement + '.txt', 'w') as f:
        f.write('Loc2Glob-GAE\n\tROC: {:.5f}±{:.5f}\tAP: {:.5f}±{:.5f}\tTime: {:.5f}±{:.5f}/epoch\n\n'.format(roc_scores[0][0], roc_scores[0][1], ap_scores[0][0], ap_scores[0][1], time_epoch[0][0], time_epoch[0][1]))
        f.write('Loc2Glob-FastGAE\n\tROC: {:.5f}±{:.5f}\tAP: {:.5f}±{:.5f}\tTime: {:.5f}±{:.5f}/epoch\n\n'.format(roc_scores[1][0], roc_scores[1][1], ap_scores[1][0], ap_scores[1][1], time_epoch[1][0], time_epoch[1][1]))
        if args.files == 'all':
            f.write('Loc2Glob-Variational FastGAE\n\tROC: {:.5f}±{:.5f}\tAP: {:.5f}±{:.5f}\tTime: {:.5f}±{:.5f}/epoch\n\n'.format(roc_scores[2][0], roc_scores[2][1], ap_scores[2][0], ap_scores[2][1], time_epoch[2][0], time_epoch[2][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset')
    parser.add_argument('--path', default=None)
    parser.add_argument('--res_state', default='', type=str)
    parser.add_argument('--files', default='all')
    parser.add_argument('--statement', default='', type=str)
    parser.add_argument('--num_seeds', default=10, type=int)
    args = parser.parse_args()
    path = 'results/' + args.dataset + args.res_state if args.path is None else args.path
    glance(path, args.statement, args.num_seeds)
