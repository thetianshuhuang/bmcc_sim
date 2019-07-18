"""Run MFM or DPM on all files in a directory.

All datasets should be saved inside a directory (possibly nested); no other
files should be inside. The results are written to RESULT_DIR, with the
directory structure mirrored.
"""

import numpy as np
from scipy.stats import poisson
import bmcc

import os
from tqdm import tqdm
from multiprocessing import Pool

from util import get_files_recursive


BASE_DIR = './data/r=0.7,sym=False'
RESULT_DIR = './results_dpm_no_eb/r=0.7,sym=False'
MIXTURE_MODEL = 'DPM'


def run(path):
    """Inner function to run DPM/MFM on a single test case"""

    path_dst = path.replace(BASE_DIR, RESULT_DIR)

    # Ignore if test already run (file present)
    if os.path.exists(path_dst):
        return

    # Load dataset
    dataset = bmcc.GaussianMixture(path, load=True)

    # Create mixture model
    if MIXTURE_MODEL == 'MFM':
        mm = bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, dataset.k))
    else:
        mm = bmcc.DPM(alpha=1, use_eb=False)

    # Initialize sampler
    model = bmcc.GibbsMixtureModel(
        data=dataset.data,
        component_model=bmcc.NormalWishart(df=dataset.d),
        mixture_model=mm,
        assignments=np.zeros(dataset.n).astype(np.uint16),
        thinning=5)

    # Run model
    try:
        for i in range(5000):
            model.iter()
            if np.max(model.assignments) > 100:
                break

        # Save
        np.savez(path_dst, hist=model.hist)

    except Exception as e:
        print("Exception in {}:".format(path))
        print(e)


if __name__ == '__main__':

    # Get list of files
    DATASETS = get_files_recursive(BASE_DIR)

    # Mirror directory structure
    for path in DATASETS:
        parent = os.path.dirname(path.replace(BASE_DIR, RESULT_DIR))
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
