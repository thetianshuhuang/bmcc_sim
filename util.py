
import os
from matplotlib import pyplot as plt


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
    """Helper function to save figure"""

    fig.set_size_inches(32, 24)
    fig.savefig(name + '.png')
    plt.close('all')
