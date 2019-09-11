import bmcc
from scipy.stats import poisson


def hybrid(*args, **kwargs):
    for _ in range(5):
        bmcc.gibbs(*args, **kwargs)
    bmcc.split_merge(*args, **kwargs)


def hybrid_3(*args, **kwargs):
    for _ in range(3):
        bmcc.gibbs(*args, **kwargs)
    bmcc.split_merge(*args, **kwargs)


def hybrid_10(*args, **kwargs):
    for _ in range(10):
        bmcc.gibbs(*args, **kwargs)
    bmcc.split_merge(*args, **kwargs)


def dpm(sampler=hybrid, alpha=1.0, eb=False):
    return {
        "sampler": sampler,
        "mixture": lambda nc: bmcc.DPM(alpha=alpha, use_eb=eb)
    }


def mfm(sampler=hybrid, offset=0, gamma=1):
    return {
        "sampler": sampler,
        "mixture": lambda nc: bmcc.MFM(
            gamma=gamma, prior=lambda k: poisson.logpmf(k, nc + offset))
    }


METHODS = {
    "dpm_gibbs": dpm(sampler=bmcc.gibbs),
    "dpm_sm": dpm(sampler=bmcc.split_merge),
    "dpm_hybrid": dpm(),
    "dpm_eb": dpm(eb=True),
    "dpm_small_alpha": dpm(alpha=0.1),
    "dpm_big_alpha": dpm(alpha=10),
    "dpm_eb_gibbs": dpm(eb=True, sampler=bmcc.gibbs),

    "mfm_gibbs": mfm(sampler=bmcc.gibbs),
    "mfm_sm": mfm(sampler=bmcc.split_merge),
    "mfm_hybrid_3": mfm(sampler=bmcc.hybrid_3),
    "mfm_hybrid_10": mfm(sampler=bmcc.hybrid_10),
    "mfm_hybrid": mfm(),
    "mfm_prior_low": mfm(offset=-2),
    "mfm_prior_high": mfm(offset=2)
}
