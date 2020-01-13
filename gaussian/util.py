
import os
from matplotlib import pyplot as plt
import gc


def get_files_recursive(path, ext=".npz"):
    """Helper function to recursively find files of a certain extension"""

    if os.path.isfile(path):
        return [path] if path.endswith(ext) else []
    else:
        res = []
        for p in os.listdir(path):
            res = res + get_files_recursive(os.path.join(path, p), ext=ext)
        return res


def save_fig(fig, name):
    """Helper function to save figure"""

    fig.set_size_inches(24, 18)
    fig.savefig(name + '.png')
    plt.close(fig)
    gc.collect()
