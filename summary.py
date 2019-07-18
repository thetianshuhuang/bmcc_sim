
import os
import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats


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
PLOT_STATS = {
    'rand': 'Rand Index',
    'nmi': 'NMI',
    'aggregation': 'Aggregation Score',
    'segregation': 'Segregation Score'
}
METHODS = {
    'dpm_noeb': 'DPM (No EB)',
    'dpm': 'DPM (With EB)',
    'mfm': 'MFM'
}


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
    targets = [f for f in os.listdir(folder) if f.endswith('.json')]

    res = [load_json(os.path.join(folder, f)) for f in targets]
    return {attr: [geterror(v, attr) for v in res] for attr in STATISTICS}


def _stats(data, attr):
    mfm, dpm, dpm_noeb = data

    return {
        "mfm": np.mean([
            i / j for i, j in zip(mfm[attr], mfm["oracle_" + attr])]),
        "dpm": np.mean([
            i / j for i, j in zip(dpm[attr], mfm["oracle_" + attr])]),
        "dpm_noeb": np.mean([
            i / j for i, j in zip(dpm_noeb[attr], mfm["oracle_" + attr])]),
        "diff": np.mean([
            i - j for i, j in zip(mfm[attr], dpm_noeb[attr])]),
        "diff_se": stats.sem([
            i - j for i, j in zip(mfm[attr], dpm_noeb[attr])])
    }


def get_stats(folder):

    mfm = load(os.path.join(MFM_DIR, folder))
    dpm = load(os.path.join(DPM_DIR, folder))
    dpm_noeb = load(os.path.join(DPM_NOEB_DIR, folder))
    data = (mfm, dpm, dpm_noeb)

    return {attr: _stats(data, attr) for attr in PLOT_STATS}


if __name__ == '__main__':

    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    targets = os.listdir(MFM_DIR)

    res = {folder: get_stats(folder) for folder in targets}

    # Summary plots
    for d in [2, 3, 5]:
        fig, axs = plt.subplots(3, 4)
        fig.suptitle(
            'Relative Performance to Oracle [d={}]'.format(d),
            fontsize=20)
        for axrow, n in zip(axs, [500, 1000, 2000]):
            for ax, stat in zip(axrow, PLOT_STATS):
                for method in METHODS:
                    data = [
                        res["n={},d={},k={}".format(n, d, k)][stat][method]
                        for k in [3, 4, 5, 6]
                    ]
                    ax.plot([3, 4, 5, 6], data, label=METHODS[method])
                    ax.set_ylim(0.3, 1.2)
                ax.legend()
                ax.set_title('{} [n={}]'.format(stat, n))

        fig.set_size_inches(24, 18)
        fig.savefig("summary_d={}".format(d))

    # Difference summary
    fig, axs = plt.subplots(3, 4)
    fig.suptitle(
        'Relative Performance of MFM, DPM (MFM - DPM)',
        fontsize=20)
    for axrow, d in zip(axs, [2, 3, 5]):
        for ax, stat in zip(axrow, PLOT_STATS):
            for n in [500, 1000, 2000]:
                data_d = [
                    res["n={},d={},k={}".format(n, d, k)][stat]
                    for k in [3, 4, 5, 6]
                ]
                data = [d["diff"] for d in data_d]
                data_err = [d["diff_se"] for d in data_d]

                ax.errorbar(
                    [3, 4, 5, 6], data, yerr=data_err,
                    label="n={}".format(n), capsize=10)
            ax.legend()
            ax.axhline(0, color='red')
            ax.set_title('{} [d={}]'.format(stat, d))

    fig.set_size_inches(24, 18)
    fig.savefig("summary_diff")
