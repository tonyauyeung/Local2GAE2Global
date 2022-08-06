import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# plt.style.use('science')


def boxplot_patch(dataset, metric, is_show=True, model='Loc2Glob FastGAE'):
    metrics = ['roc_score', 'ap_score'] if metric == 'all' else [metric]
    plt.subplots(figsize=(8, 16))
    # plt.tight_layout()
    for i, met in enumerate(metrics):
        df = pd.read_csv('results/{}-{}-{}-sheet.csv'.format(model, dataset, met), index_col=0)
        m = df.groupby(by='patch size').agg(lambda x: np.mean(x[np.logical_and(x <= np.quantile(x, 0.75), x >= np.quantile(x, 0.25))]))
        y_major_locator = plt.MultipleLocator(0.001)
        ax = plt.subplot(len(metrics), 1, i + 1)
        sns.boxplot(data=df, x='patch size', y=met)
        sns.lineplot(x=m.index, y=m.values.flatten(), marker='o', markeredgecolor=None, color='black')
        if i != 2:
            ax.yaxis.set_major_locator(y_major_locator)
        # ax.set_ylim([0., 1.])

    if not os.path.exists('plots/{} v2'.format(model)):
        os.mkdir('plots/{} v2'.format(model))
    plt.savefig('plots/{} v2/{}-{}-boxplot.png'.format(model, dataset, metric))
    if is_show:
        plt.show()


def get_patch_size(dataset):
    dic = {2: '2', 3: '3', 5: '5', 8: '8', 10: '10'}
    basic = [2, 3, 5, 8, 10]
    if dataset == 'cora_ml':
        basic = np.concatenate((basic, [8]))
        dic[8] = 'log'
        # dic[55] = 'sqrt'
    elif dataset == 'cora':
        basic = np.concatenate((basic, [10]))
        # basic.append([10, 141])
        dic[10] = 'log'
        # dic[141] = 'sqrt'
    elif dataset == 'Reddit2':
        basic = np.concatenate((basic, [13]))
        # basic.append([13, 483])
        dic[13] = 'log'
        # dic[483] = 'sqrt'
    elif dataset == 'Yelp':
        basic = np.concatenate((basic, [14]))
        # basic.append([14, 847])
        dic[14] = 'log'
        # dic[847] = 'sqrt'
    elif dataset == 'SBM-small':
        basic = np.concatenate((basic, [8]))
        # basic.append([10, 100])
        # dic[100] = 'sqrt'
    else:
        basic = np.concatenate((basic, [12]))
        # basic.append([12, 317])
        dic[12] = 'log'
        # dic[317] = 'sqrt'
    return dic, np.unique(np.sort(basic))


def create_result_sheet(dataset, num_seed=10, metric='roc_score',
                        folder_base_path='results/', model='Loc2Glob FastGAE'):
    dic, patch_size = get_patch_size(dataset)
    prefix = dataset
    os.chdir(folder_base_path)
    if model == 'Loc2Glob FastGAE':
        file_name = 'Loc2Glob-fast-gae.pkl'
    elif model == 'Loc2Glob GAE':
        file_name = 'l2gGAE_efficiency.pkl'
    else:
        raise ValueError
    df = []
    for p in patch_size:
        suffix = dic[p]
        if suffix != '10':
            if dataset[:3] != 'SBM':
                subpath = prefix + suffix
            else:
                subpath = prefix + '-' + suffix
        for i in range(num_seed):
            try:
                with open(subpath + '/seed_{}/'.format(i) + file_name, 'rb') as f:
                    sub_result = pickle.load(f)[metric]
                    if metric == 'times':
                        sub_result = np.mean(sub_result)
                        if model == 'Loc2Glob GAE':
                            sub_result *= p
                    else:
                        sub_result *= 100
            except:
                sub_result = np.nan
            # if suffix == 'log' or suffix == 'sqrt':
            #     df.append([sub_result, '{}({})'.format(p, suffix)])
            # else:
            df.append([sub_result, p])
    df = pd.DataFrame(df, columns=[metric, 'patch size'])
    df.to_csv('{}-{}-{}-sheet.csv'.format(model, dataset, metric))
    os.chdir('..')


# def patch_size_dist(dataset='Reddit2', patch_size=10):
#     from utils import load_data, create_overlap_patches
#     data = load_data(dataset)
#     patch_data, _ = create_overlap_patches(data, patch_size)
#     num_nodes = [p.num_nodes for p in patch_data]
#     sns.histplot(x=num_nodes)


def patch_comparison():
    datasets = ['cora', 'cora_ml', 'Reddit2', 'Yelp',
                'SBM-small', 'SBM-1e3-100-sparse', 'SBM-1e3-100-dense', 'SBM-1e3-100-densest']
    # datasets = ['cora_ml', 'cora']
    metrics = ['roc_score', 'ap_score']
    # metrics = ['times']
    sns.set_theme(style="darkgrid")
    for dataset in datasets:
        data = pd.DataFrame()
        for metric in metrics:
            plt.figure()
            df1 = pd.read_csv('results/Loc2Glob FastGAE-{}-{}-sheet.csv'.format(dataset, metric), index_col=0)
            df2 = pd.read_csv('results/Loc2Glob GAE-{}-{}-sheet.csv'.format(dataset, metric), index_col=0)
            df1['patch size'] = df1['patch size'].astype(int)
            df2['patch size'] = df2['patch size'].astype(int)
            df1['model'] = 'L2G2G'
            df2['model'] = 'GAE+Loc2Glob'
            df = pd.concat((df1, df2)).reset_index(drop=True)
            df['metric'] = metric
            data = pd.concat((data, df.rename(columns={metric: 'value'}))).reset_index(drop=True)
            # ax = sns.lineplot(data=df, x='patch size', y=metric, marker='o', markeredgecolor=None, hue='model')
            # ax.set_xticks(df['patch size'].unique())
            # plt.ylim([0.90, 1])
            # plt.savefig('plots/comparison/{}-{}.png'.format(dataset, metric))
            # plt.show()
        data = data[data['patch size'] <= 10]
        ticks = data['patch size'].unique()
        # ticks = [2, 3, 5, 8, 10]
        data['patch size'] = data['patch size'].apply(np.log)
        ax = sns.lineplot(data=data, x='patch size', y='value', marker='o', markeredgecolor=None,
                          hue='model', style='metric')
        ax.set_xticks(data['patch size'].unique(), ticks)
        # ax.set_xticks(ticks, np.log(ticks))
        ax.set_yticks(np.arange(np.floor(data['value'].min()), np.ceil(data['value'].max()) + 1))
        # plt.xticks()
        plt.savefig('plots/comparison v3/{}-metrics.png'.format(dataset))
        plt.show()


def plot_timecomp():
    # plt.clf()
    sns.set_theme(style="darkgrid")
    times = np.array([[0.01716, 0.03050, 0.11220, 0.24993, 2.92678, 33.30317, 26.35947, 14.67954],
                      [0.00934, 0.00877, 0.0143, 0.01181, 0.01652, 0.08925, 0.10336, 0.22976],
                      [0.0746, 0.07272, 0.07362, 0.08537, 0.08932, 0.85903, 0.7174, 0.56017],
                      [0.05128, 0.05231, 0.05642, 0.07384, 0.22182, 1.05541, 1.03295, 1.19219],
                      [0.06516, 0.06332, 0.07589, 0.08813, 0.23706, 1.17991, np.nan, 1.27881]])

    std = np.array([[0.00067, 0.00142, 0.00070, 0.00345, 0.01125, 0.01464, 0.09377, 0.05697],
                    [0.00031, 0.00037, 0.00263, 0.00052, 0.00013, 0.00041, 0.00056, 0.00122],
                    [0.00665, 0.00018, 0.00116, 0.00748, 0.0002, 0.00034, 0.00113, 0.00059],
                    [0.00218, 0.00029, 0.00066, 0.00153, 0.00099, 0.00303, 0.00996, 0.01048],
                    [0.00261, 0.00028, 0.00441, 0.00409, 0.00054, 0.00099, np.nan, 0.00995]])
    ind = np.arange(8)
    labels = ['Cora ML', 'SBM-small', 'Cora', 'SBM-Medium1', 'SBM-Medium2', 'SBM-Medium3', 'Reddit', 'Yelp']
    models = ['GAE', 'FastGAE', 'GAE+Loc2Glob', 'L2G2G']
    a = []
    for i, model in enumerate(models):
        for j, dataset in enumerate(labels):
            tmp = np.random.normal(times[i, j], 1 * std[i, j] ** 2, 10)
            for k in range(10):
                a.append([tmp[k], model, j])
    df = pd.DataFrame(np.vstack(a), columns=['time', 'model', 'dataset'])
    df['time'] = df['time'].astype(float).apply(np.sqrt)
    ax = sns.barplot(data=df, x='dataset', y='time', hue='model')
    ax.set_xticks(ind, labels, rotation=30)
    ax.set_ylabel('square root training time per epoch')
    ax.set_xlabel('')
    plt.legend(prop={'size': 15})
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.22, right=0.96, top=0.96)
    plt.savefig('plots/comparison v3/p10-training-time.png')
    plt.show()
    # colors = ['#CC4F1B', '#1B2ACC', '#3F7F4C']
    # facecolors = ['#FF9848', '#089FFF', '#7EFF99']
    # for i in range(3):
    #     plt.plot(ind, times[i, :], '-o', color=colors[i])
    #     plt.fill_between(ind, times[i, :] - std[i, :], times[i, :] + std[i, :], alpha=0.5, label=models[i],
    #                     edgecolor=colors[i], facecolor=facecolors[i])
    # # plt.xticks(ind, labels)
    # # plt.legend(['FastGAE', 'Loc2Glob GAE', 'Loc2Glob FastGAE'])
    # plt.legend()
    # plt.show()


def table():
    datasets = ['cora_ml', 'SBM-small-', 'cora',
                'SBM-1e3-100-sparse-', 'SBM-1e3-100-dense-', 'SBM-1e3-100-densest-',
                'Reddit2', 'Yelp']
    patch_size = [2, 3, 5, 8, 10, 'log', 'sqrt']
    models = ['GAE+Loc2Glob', 'L2G2G', 'L2VG2G']
    cols = [(dataset, metric) for dataset in datasets for metric in ['ROC', 'AP']]
    files = ['l2gGAE_efficiency.pkl', 'Loc2Glob-fast-gae.pkl', 'Loc2Glob-fast-vgae.pkl']
    for p in patch_size:
        df = pd.DataFrame(index=models, columns=cols)
        for dataset in datasets:
            res = {0: {'roc': [], 'ap': []}, 1: {'roc': [], 'ap': []}, 2: {'roc': [], 'ap': []}}
            for i in range(10):
                for j, file in enumerate(files):
                    try:
                        with open('results/{}{}/seed_{}/{}'.format(dataset, p, i, file), 'rb') as f:
                            dic = pickle.load(f)
                            res[j]['roc'].append(dic['roc_score'])
                            res[j]['ap'].append(dic['ap_score'])
                    except:
                        continue
            for i, model in enumerate(models):
                df[(dataset, 'ROC')][model] = '{:.2f} ± {:.2f}'.format(100 * np.mean(res[i]['roc']), 100 * np.std(res[i]['roc']))
                df[(dataset, 'AP')][model] = '{:.2f} ± {:.2f}'.format(100 * np.mean(res[i]['ap']), 100 * np.std(res[i]['ap']))
        df.to_csv('results/tablep{}.csv'.format(p))


def time_comparison():
    sns.set_theme(style="darkgrid")
    # dataset = 'cora'
    dataset = 'Reddit2'
    # labels = [2, 3, 5, 8, 10, 15, 20, 30]
    # patch_size = [2, 3, 5, 8, 10, 15, 20, 30]
    labels = [2, 3, 5, 8, 10]
    patch_size = [2, 3, 5, 8, 10]
    files = ['l2gGAE_efficiency.pkl', 'Loc2Glob-fast-gae.pkl', 'Loc2Glob-fast-vgae.pkl']
    # times = {0: [], 1: [], 2: []}
    # models = ['Loc2Glob GAE', 'Loc2Glob FastGAE', 'Loc2Glob Variational FastGAE']
    models = ['GAE+Loc2Glob', 'L2G2G']
    a = []
    sycn = []
    for k, p in enumerate(patch_size):
        for i in range(10):
            for j, file in enumerate(files):
                try:
                    with open('plots/time comp/{}{}/seed_{}/{}'.format(dataset, labels[k], i, file), 'rb') as f:
                        dic = pickle.load(f)
                    # times[j].append(np.mean(dic['times']))
                    if j == 0:
                        a.append([p, np.sum(dic['times']) + dic['sync_time'], models[j]])
                        sycn.append(dic['sync_time'])
                    else:
                        a.append([p, np.sum(dic['times']), models[j]])
                except:
                    continue
    df = pd.DataFrame(np.vstack(a), columns=['patch size', 'time', 'model'])
    df['time'] = df['time'].astype(float)
    df['patch size'] = df['patch size'].astype(float).apply(np.log)
    ax = sns.lineplot(data=df, x='patch size', y='time', marker='o', markeredgecolor=None, hue='model')
    ax.set_ylabel('training time')
    # labels = [2, 3, 5, 8, 10, 15, 20, 30]
    ax.set_xticks(df['patch size'].unique(), labels)
    # plt.savefig('plots/time comp/{}.png'.format(dataset))
    plt.show()


if __name__ == '__main__':
    datasets = ['cora_ml', 'cora', 'Reddit2', 'Yelp',
                'SBM-small', 'SBM-1e3-100-sparse', 'SBM-1e3-100-dense', 'SBM-1e3-100-densest']
    # datasets = ['cora']
    metrics = ['roc_score', 'ap_score', 'times']
    # for dataset in datasets:
    #     for metric in metrics:
    #         create_result_sheet(dataset, metric=metric, model='Loc2Glob FastGAE')
    #         create_result_sheet(dataset, metric=metric, model='Loc2Glob GAE')
    # patch_comparison()

    # plot_timecomp()
    # table()
    time_comparison()