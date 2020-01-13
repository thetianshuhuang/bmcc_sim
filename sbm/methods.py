import bmcc
from scipy.stats import poisson


def dpm(alpha=1.0, eb=False):
    return {
        "sampler": bmcc.gibbs,
        "mixture": lambda nc: bmcc.DPM(alpha=alpha, use_eb=eb)
    }


def mfm(offset=0, gamma=1):
    return {
        "sampler": bmcc.gibbs,
        "mixture": lambda nc: bmcc.MFM(
            gamma=gamma, prior=lambda k: poisson.logpmf(k, nc + offset))
    }


METHODS = {
    "dpm": dpm(),
    "dpm_small_alpha": dpm(alpha=0.1),
    "dpm_big_alpha": dpm(alpha=10),
    "mfm": mfm(),
    "mfm_prior_low": mfm(offset=-1),
    "mfm_prior_high": mfm(offset=1)
}
