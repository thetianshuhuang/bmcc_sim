
import numpy as np
import bmcc

import os
from tqdm import tqdm
from multiprocessing import Pool

from util import get_files_recursive
from methods import METHODS


BASE_DIR = "./data"
RESULT_DIR = "./results"


def run(args):

    # Unpack
    path, method_name = args

    dst = path.replace(
        BASE_DIR, os.path.join(RESULT_DIR, method_name))

    # Ignore if test already run (file present)
    if os.path.exists(dst):
        return

    # Load dataset
    dataset = bmcc.GaussianMixture(path, load=True)

    # Fetch method
    method = METHODS[method_name]

    # Create model
    model = bmcc.BayesianMixture(
        data=dataset.data,
        sampler=method["sampler"],
        component_model=bmcc.NormalWishart(df=dataset.d),
        mixture_model=method["mixture"](dataset.k),
        assignments=np.zeros(dataset.n).astype(np.uint16),
        thinning=5)

    # Run iterations (break on exceeding limit)
    try:
        for i in range(5000):
            model.iter()
            if np.max(model.assignments) > 100:
                break
    except Exception as e:
        print("Exception in {} / {}:".format(method_name, path))
        print(e)

    np.savez(dst, hist=model.hist)


if __name__ == '__main__':

    # Get list of files
    DATASETS = get_files_recursive(BASE_DIR)

    # Mirror directory structure
    for path in DATASETS:
        for method_name in METHODS:
            parent = os.path.dirname(
                path.replace(
                    BASE_DIR,
                    os.path.join(RESULT_DIR, method_name)))

            if not os.path.exists(parent):
                os.makedirs(parent)

    # Take all combinations of datasets, methods
    TESTS = [(ds, method) for ds in DATASETS for method in METHODS]

    p = Pool(processes=1)
    with tqdm(total=len(TESTS)) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(run, TESTS))):
            pbar.update()
