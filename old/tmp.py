import numpy as np
import json

import bmcc

import os
from tqdm import tqdm
from multiprocessing import Pool

from util import get_files_recursive


BASE_DIR = './data/r=0.7,sym=False'
RESULT_DIR = './results_mfm/r=0.7,sym=False'
PLOT_DIR = './plots_mfm/r=0.7,sym=False'
MIXTURE_MODEL = 'MFM'


def run(path):

    path_dst = path.replace(BASE_DIR, RESULT_DIR)
    path_plot = path.replace(BASE_DIR, PLOT_DIR)

    # Load dataset
    dataset = bmcc.GaussianMixture(path, load=True)
    hist = np.load(path_dst)['hist']

    # Check for cut off
    if(hist.shape[0] < 500):
        return

    with open(path_plot + '_scores.json', 'r') as f:
        scores = json.loads(f.read())

    asn = dataset.assignments
    scores.update({
        "aggregation": bmcc.aggregation_score(asn, hist[scores["best_idx"]]),
        "segregation": bmcc.segregation_score(asn, hist[scores["best_idx"]]),
        "oracle_aggregation": bmcc.aggregation_score(asn, dataset.oracle),
        "oracle_segregation": bmcc.aggregation_score(asn, dataset.oracle)
    })

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
