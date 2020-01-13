
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

# Limit to 100 clusters
CLUSTERS_LIMIT = 100


def save_fig(fig, name):
    """Helper function to save figure"""

    fig.set_size_inches(16, 12)
    fig.savefig(name + '.png')
    fig.clf()
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
    dataset = bmcc.StochasticBlockModel(path, load=True)

    # Fetch method
    method = METHODS[method_name]

    cm = bmcc.SBM(a=1, b=1)
    mm = method["mixture"](dataset.k)

    # Create model
    model = bmcc.BayesianMixture(
        data=dataset.data,
        sampler=method["sampler"],
        component_model=cm,
        mixture_model=mm,
        assignments=np.zeros(dataset.n).astype(np.uint16),
        thinning=5)

    # Run iterations (break on exceeding limit)
    try:
        for i in range(1000):
            model.iter()
            if np.max(model.assignments) > CLUSTERS_LIMIT:
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
    if os.path.exists(eval_dir + "_scores.json"):
        return

    # Load dataset
    dataset = bmcc.StochasticBlockModel(path, load=True)
    hist = np.load(result_dir)['hist']

    if hist.shape[0] < 2:
        with open(eval_dir + "_scores.json", "w") as f:
            f.write(json.dumps({"errored": "no iterations saved"}))
        return

    # If procedure terminates before 2000it (400 once thinned), use 2nd half of
    # samples
    bi_base = min(int(hist.shape[0] / 2), 200)

    # Base evaluation
    # We don't care about the oracle matrix for now, so skip computation
    res = bmcc.LstsqResult(dataset.data, hist, burn_in=bi_base)
    res.evaluate(dataset.assignments)

    # save trace
    TRACE_PLOTS = {
        "Metrics (NMI, Rand)": {
            "Normalized Mutual Information": "nmi",
            "Adjusted Rand Index": "rand",
        },
        "Number of Clusters": {
            "Number of Clusters": "num_clusters",
            "Actual": "clusters_true"
        },
        "Aggregation / Segregation Score": {
            "Aggregation Score": "aggregation",
            "Segregation Score": "segregation",
        }
    }
    save_fig(res.trace(plots=TRACE_PLOTS), eval_dir + "_trace")

    # Save scores
    scores = {
        "rand": res.rand_best,
        "nmi": res.nmi_best,
        "num_clusters": int(res.num_clusters[res.best_idx]),
        "best_idx": int(res.best_idx),
        "aggregation": res.aggregation_best,
        "segregation": res.segregation_best,
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

    # Take all combinations of datasets, methods
    TESTS = [(ds, method) for ds in DATASETS for method in METHODS]

    # Mirror directory structure
    for path, method_name in TESTS:
        parent = os.path.dirname(
            path.replace(BASE_DIR, os.path.join(bd, method_name)))

        if not os.path.exists(parent):
            os.makedirs(parent)

    if "single" in sys.argv:
        # Run - single threaded (for debugging)
        for t in tqdm(TESTS):
            rf(t)
    else:
        # Run - parallel (for deployment)
        p = Pool()
        with tqdm(total=len(TESTS)) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(rf, TESTS))):
                pbar.update()
