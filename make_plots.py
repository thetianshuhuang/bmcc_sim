import numpy as np
from matplotlib import pyplot as plt
import bmcc
import json

import os
from tqdm import tqdm
from multiprocessing import Pool


BASE_DIR = './data/r=0.7,sym=False'
RESULT_DIR = './results_dpm_no_eb/r=0.7,sym=False'
PLOT_DIR = './plots_dpm_no_eb/r=0.7,sym=False'
MIXTURE_MODEL = 'MFM'


def get_files_recursive(path):
    """Helper function to recursively find .npz files"""

    if os.path.isfile(path):
        return [path] if path.endswith(".npz") else []
    else:
        res = []
        for p in os.listdir(path):
            res = res + get_files_recursive(os.path.join(path, p))
        return res


def save_fig(fig, name):
    fig.set_size_inches(32, 24)
    fig.savefig(name + '.png')
    plt.close('all')


def run(path):

    path_dst = path.replace(BASE_DIR, RESULT_DIR)
    path_plot = path.replace(BASE_DIR, PLOT_DIR)

    # Load dataset
    dataset = bmcc.GaussianMixture(path, load=True)
    hist = np.load(path_dst)['hist']

    # Check for cut off
    if(hist.shape[0] < 500):
        with open(path_plot + '_scores.json', 'w') as f:
            f.write(json.dumps({"errored": True}))
        return

    # Result
    res = bmcc.LstsqResult(dataset.data, hist, burn_in=400)
    res.evaluate(
        dataset.assignments,
        oracle=dataset.oracle,
        oracle_matrix=dataset.oracle_matrix)

    save_fig(res.trace(), path_plot + '_trace')
    save_fig(res.matrices(), path_plot + '_mat')
    save_fig(
        res.clustering(kwargs_scatter={"marker": "."}),
        path_plot + '_cluster')

    save_fig(
        dataset.plot_oracle(), path_plot + '_oracle')

    scores = {
        "rand": res.rand_best,
        "nmi": res.nmi_best,
        "oracle_rand": res.oracle_rand,
        "oracle_nmi": res.oracle_nmi,
        "num_clusters": int(res.num_clusters[res.best_idx]),
        "best_idx": int(res.best_idx)
    }

    with open(path_plot + '_scores.json', 'w') as f:
        f.write(json.dumps(scores))


if __name__ == '__main__':

    # Get list of files
    DATASETS = get_files_recursive(BASE_DIR)

    # Mirror directory structure
    for path in DATASETS:
        parent = os.path.dirname(path.replace(BASE_DIR, PLOT_DIR))
        if not os.path.exists(parent):
            os.makedirs(parent)

    # Info
    print('Found {} files:'.format(len(DATASETS)))
    print('\n'.join(DATASETS[:3]))
    if len(DATASETS) > 3:
        print('({} more) ...'.format(len(DATASETS) - 3))

    print('\n')
    print('Starting...')

    # Run pool; tqdm is used to provide a progress bar
    p = Pool()
    with tqdm(total=len(DATASETS)) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(run, DATASETS))):
            pbar.update()
