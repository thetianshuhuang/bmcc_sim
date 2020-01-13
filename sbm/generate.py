

import os
import bmcc
import uuid
import numpy as np
from tqdm import tqdm


def get_dirname(n, k, p, r, base_dir="./data"):
    target = os.path.join(
        base_dir,
        "n={},p={},k={},r={}".format(n, k, p, r))
    if not os.path.isdir(target):
        os.makedirs(target)
    return target


def make(n, k, p, r):
    print("Making: n={}, k={}, p={}, r={}".format(n, k, p, r))

    for _ in tqdm(range(100)):
        dst = get_dirname(n, k, p, r)
        Q = np.ones((k, k)) * 0.1 + np.identity(k) * (p - 0.1)
        ds = bmcc.StochasticBlockModel(
            n=n, k=k, r=r, shuffle=False, Q=Q)
        ds.save(os.path.join(dst, str(uuid.uuid4())))


def make_phase_1():
    for n in [200, 500, 1000]:
        for k in [3, 5, 8]:
            make(n, k, 0.3, 1)


def make_phase_2():
    # 0.3 is covered by phase 1
    for p in [0.2, 0.25, 0.35, 0.4, 0.45, 0.5]:
        make(500, 3, p, 1)


if __name__ == '__main__':
    import sys

    if sys.argv[1] == '1':
        make_phase_1()
    elif sys.argv[1] == '2':
        make_phase_2()
    else:
        print("Specify the phase to run.")
        exit()
