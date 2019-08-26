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

        for _ in tqdm(range(100)):

            ds = bmcc.GaussianMixture(
                n=80 * d * k, k=k, d=d, r=0.8, alpha=40, df=d,
                symmetric=False, shuffle=False)
            ds.save(os.path.join(dst, str(uuid.uuid4())))


if __name__ == '__main__':
    make_phase_1()
