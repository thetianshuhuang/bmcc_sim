
import numpy as np
import bmcc

import os
import sys
from tqdm import tqdm
from multiprocessing import Pool
import json

from matplotlib import pyplot as plt
import gc

from util import get_files_recursive
from methods import METHODS


BASE_DIR = "./data"
RESULT_DIR = "./results"
EVAL_DIR = "./eval"


def save_fig(fig, name):
    """Helper function to save figure"""

    fig.set_size_inches(16, 12)
    fig.savefig(name + '.png')
    plt.close(fig)
    gc.collect()


def run_sample(args):
    """Run MCMC sampling"""

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


def run_evaluate(args):
    """Evaluate samples"""

    path, method_name = args

    result_dir = path.replace(BASE_DIR, os.path.join(RESULT_DIR, method_name))
    eval_dir = path.replace(BASE_DIR, os.path.join(EVAL_DIR, method_name))

    # Ignore if already run
    if os.path.exists(eval_dir):
        return

    # Load dataset
    dataset = bmcc.GuassianMixture(path, load=True)
    hist = np.load(result_dir)['hist']

    # If procedure terminates before 2000it, use 2nd half of samples
    bi_base = min(int(hist.shape[0] / 2), 1000)

    # Base evaluation
    # We don't care about the oracle matrix for now, so skip computation
    res = bmcc.LstsqResult(dataset.data, hist, burn_in=bi_base)
    res.evaluate(
        dataset.assignments,
        oracle=dataset.oracle,
        oracle_matrix=None)

    # save trace
    save_fig(res.trace(), eval_dir + "_trace")

    # Save scores
    scores = {
        "rand": res.rand_best,
        "nmi": res.nmi_best,
        "oracle_rand": res.oracle_rand,
        "oracle_nmi": res.oracle_nmi,
        "num_clusters": int(res.num_clusters[res.best_idx]),
        "best_idx": int(res.best_idx),
        "aggregation": res.aggregation_best,
        "segregation": res.segregation_best,
        "oracle_aggregation": res.oracle_aggregation,
        "oracle_segregation": res.oracle_segregation,
        "iterations": hist.shape[0]
    }
    with open(eval_dir + "_scores.json", "w") as f:
        f.write(json.dumps(scores))


if __name__ == '__main__':

    if sys.argv[1] == 'sample':
        rf = run_sample
        bd = RESULT_DIR
    elif sys.argv[1] == 'eval':
        rf = run_evaluate
        bd = EVAL_DIR
    else:
        print("Specify 'sample' or 'eval'.")
        exit()

    # Get list of files
    DATASETS = get_files_recursive(BASE_DIR)

    # Mirror directory structure
    for path in DATASETS:
        for method_name in METHODS:
            parent = os.path.dirname(
                path.replace(BASE_DIR, os.path.join(bd, method_name)))

            if not os.path.exists(parent):
                os.makedirs(parent)

    # Take all combinations of datasets, methods
    TESTS = [(ds, method) for ds in DATASETS for method in METHODS]

    p = Pool()
    with tqdm(total=len(TESTS)) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(rf, TESTS))):
            pbar.update()
