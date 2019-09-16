import matplotlib


def setup_latex(label_size=8, title_size=10):
    matplotlib.rcParams.update({
        'backend': 'ps',
        'axes.labelsize': label_size,
        'axes.titlesize': title_size,
        'legend.fontsize': label_size,
        'xtick.labelsize': label_size,
        'ytick.labelsize': label_size,
        'text.usetex': True,
        'font.family': 'serif'
    })
