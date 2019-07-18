
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import numpy as np


MFM_DIR = './plots_mfm/r=0.7,sym=False'
DPM_DIR = './plots_dpm/r=0.7,sym=False'
DPM_NOEB_DIR = './plots_dpm_no_eb/r=0.7,sym=False'
RES_DIR = './aggregate/r=0.7,sym=False'
STATISTICS = [
    'oracle_nmi',
    'oracle_rand',
    'oracle_segregation',
    'oracle_aggregation',
    'nmi',
    'rand',
    'segregation',
    'aggregation',
    'num_clusters'
]


def load_json(fname):

    with open(fname) as file:
        return json.loads(file.read())


def attr(d, attr):
    return [v[attr] for k, v in d]


def geterror(d, k):
    if k in d:
        return d[k]
    else:
        return -0.01


def load(folder):
    targets = [
        f for f in os.listdir(folder)
        if f.endswith('.json')
    ]
    res = [load_json(os.path.join(folder, f)) for f in targets]
    return {attr: [geterror(v, attr) for v in res] for attr in STATISTICS}


def minnz(x):
    return min(i for i in x if i > 0)


def plot_scatter(ax, data, attr, name):

    (mfm, dpm, dpmn) = data

    lim = 0.9 * min(
        minnz(mfm['oracle_' + attr]),
        minnz(mfm[attr]),
        minnz(dpm[attr]),
        minnz(dpmn[attr]))

    mfma = [i if i >= 0 else lim for i in mfm[attr]]
    dpma = [i if i >= 0 else lim for i in dpm[attr]]
    dpmna = [i if i >= 0 else lim for i in dpmn[attr]]

    ax.plot(mfm['oracle_' + attr], mfma, 'o', label='MFM')
    ax.set_xlim(lim, 1.05)
    ax.set_ylim(lim, 1.05)
    ax.plot(mfm['oracle_' + attr], dpma, 'o', label='DPM (EB)')
    ax.plot(mfm['oracle_' + attr], dpmna, 'o', label='DPM (No EB)')
    ax.plot([lim, lim], [1, 1], '-')
    ax.set_title(name)
    ax.set_xlabel('Oracle ' + name)
    ax.set_ylabel('Least Squares ' + name)
    ax.legend()


def plot_hist(ax, data, attr, name):

    (mfm, dpm, dpm_noeb) = data

    d1 = [i - j for i, j in zip(mfm[attr], dpm[attr])]
    d2 = [i - j for i, j in zip(mfm[attr], dpm_noeb[attr])]
    left = min(min(d1), min(d2))
    right = max(max(d1), max(d2))
    ls = np.linspace(left, right, 100)
    bins = np.linspace(left, right, 20)

    ax.hist(d1, bins=bins, label='MFM - DPM (With EB)', alpha=0.5)
    ax.plot(ls, gaussian_kde(d1).evaluate(ls), label='MFM - DPM (With EB)')
    ax.set_title(name + ' Difference')
    ax.set_xlabel('MFM {n} - DPM {n}'.format(n=name))
    ax.set_xlim(left, right)

    ax.hist(d2, bins=bins, label='MFM - DPM (No EB)', alpha=0.5)
    ax.plot(ls, gaussian_kde(d2).evaluate(ls), label='MFM - DPM (No EB)')
    ax.set_title(name + ' Difference')
    ax.set_xlabel('MFM {n} - DPM {n}'.format(n=name))
    ax.set_xlim(left, right)

    ax.legend()


def make_summary(folder):

    mfm = load(os.path.join(MFM_DIR, folder))
    dpm = load(os.path.join(DPM_DIR, folder))
    dpm_noeb = load(os.path.join(DPM_NOEB_DIR, folder))
    data = (mfm, dpm, dpm_noeb)

    fig, axs = plt.subplots(2, 4)

    plot_scatter(axs[0][0], data, 'nmi', 'NMI')
    plot_scatter(axs[0][1], data, 'rand', 'Rand Index')
    plot_scatter(axs[0][2], data, 'aggregation', 'Aggregation Score')
    plot_scatter(axs[0][3], data, 'segregation', 'Segregation Score')

    plot_hist(axs[1][0], data, 'nmi', 'NMI')
    plot_hist(axs[1][1], data, 'rand', 'Rand Index')
    plot_hist(axs[1][2], data, 'aggregation', 'Aggregation Score')
    plot_hist(axs[1][3], data, 'segregation', 'Segregation Score')

    fig.set_size_inches(24, 12)
    fig.savefig(os.path.join(RES_DIR, folder + '.png'))
    plt.close('all')


if __name__ == '__main__':

    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    targets = os.listdir(MFM_DIR)

    p = Pool()
    with tqdm(total=len(targets)) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(make_summary, targets))):
            pbar.update()
