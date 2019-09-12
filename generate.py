"""
Phase 1: Comparison

Tests [100 replicates]
----------------------
- d,k = (3, 3), (3, 5), (3, 8), (5, 3), (8, 3)
- p=80 (n=80 * d * k)
- r=0.8
- alpha=40

Test Algorithms
---------------
[DPM] DPM; alpha=1
[DPM_0.1] DPM; alpha=0.1
[DPM_EB] DPM with EB

[MFM_k-1] MFM; prior=poisson(k - 1); gamma=1
[MFM_k] MFM; prior=poisson(k); gamma=1
[MFM_k+1] MFM; prior=poisson(k + 1); gamma=1

[DPM_SM] DPM; alpha=1; Split Merge only
[MFM_SM] MFM; prior=poisson(k); gamma=1; Split Merge only
[DPM_5_1] DPM; alpha=1; 5,1 split merge scheme
[MFM_5_1] MFM; prior=poisson(k); gamma=1; 5,1 split merge scheme

All with and without cleanup
"""

import os
import bmcc
import uuid
import numpy as np
from tqdm import tqdm


def get_dirname(d, p, k, r, base_dir="./data", makedir=False):

    target = os.path.join(
        base_dir,
        "d={},p={},k={},r={}".format(d, p, k, r))

    if not os.path.isdir(target):
        os.makedirs(target)

    return target


def make_phase_1():

    for d, k in [(3, 3), (3, 5), (3, 8), (5, 3), (8, 3)]:
        dst = get_dirname(d, 80 * d * k, k, 0.8, makedir=True)

        print(dst)
        for _ in tqdm(range(100)):

            ds = bmcc.GaussianMixture(
                n=80 * d * k, k=k, d=d, r=0.8, alpha=40, df=d,
                symmetric=False, shuffle=False)
            ds.save(os.path.join(dst, str(uuid.uuid4())))


def make_phase_2():

    for d in [3, 4, 5, 6, 8, 10, 12, 15, 18, 21]:
        for n in [600, 800, 1000]:

            dst = get_dirname(d, n, 3, 1.0, makedir=True)

            print(dst)
            for _ in tqdm(range(100)):

                ds = bmcc.GaussianMixture(
                    n=n, k=3, d=d, r=1.0, alpha=40, df=d,
                    symmetric=False, shuffle=False)
                ds.save(os.path.join(dst, str(uuid.uuid4())))


def make_phase_3():

    for k in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]:

        dst = get_dirname(3, 50 * k, k, 1.0, makedir=True)

        print(dst)
        for _ in tqdm(range(100)):
            means = [
                k * np.random.uniform(low=-0.5, high=0.5, size=3)
                for _ in range(k)
            ]

            ds = bmcc.GaussianMixture(
                n=50 * k, k=k, d=3, r=1.0, df=3,
                symmetric=False, shuffle=False, means=means)
            ds.save(os.path.join(dst, str(uuid.uuid4())))


if __name__ == '__main__':
    import sys

    if sys.argv[1] == '1':
        make_phase_1()

    elif sys.argv[1] == '2':
        make_phase_2()

    elif sys.argv[1] == '3':
        make_phase_3()

    else:
        print("Specify the phase to run.")
        exit()
