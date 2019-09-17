"""Various definitions."""

import numpy as np


COLUMNS = {
    'rand': np.float64,
    'nmi': np.float64,
    'oracle_rand': np.float64,
    'oracle_nmi': np.float64,
    'num_clusters': np.int16,
    'best_idx': np.int16,
    'aggregation': np.float64,
    'segregation': np.float64,
    'oracle_aggregation': np.float64,
    'oracle_segregation': np.float64,
    'iterations': np.int16,
    'method': str,
    'd': np.int16,
    'p': np.int16,
    'k': np.int16,
    'r': np.float64,
    'id': str,
    'file': str,
    'phase': str,
}


SCORES = {
    'rand': "Rand Score",
    'nmi': "NMI",
    'aggregation': "Aggregation Score",
    'segregation': "Segregation Score"
}


METHODS_P1 = {
    'mfm_prior_high': "MFM: poisson$(k + 2)$",
    'mfm_hybrid': "MFM: (5,1) hybrid",
    'mfm_prior_low': "MFM: poisson$(k - 2)$",
    'mfm_gibbs': "MFM: gibbs",
    'mfm_sm': "MFM: split merge",
    'mfm_hybrid_3': "MFM: (3,1) hybrid",
    'mfm_hybrid_10': "MFM: (10,1) hybrid",
    'dpm_small_alpha': r"DPM: $\alpha=0.1$",
    'dpm_hybrid': "DPM: (5,1) hybrid",
    'dpm_big_alpha': r"DPM: $\alpha=10$",
    'dpm_sm': "DPM: split merge",
    'dpm_gibbs': "DPM: gibbs",
    'dpm_eb': "DPM: EB",
    'dpm_eb_gibbs': "DPM: EB, gibbs"
}


METHODS_P2 = {
    'mfm_hybrid': "MFM",
    'dpm_hybrid': "DPM",
}


TESTS_P1 = [
    {'d': d, 'k': k, 'p': 80 * d * k, 'r': 0.8}
    for d, k, in [(3, 3), (3, 5), (3, 8), (5, 3), (8, 3)]
]


TESTS_P2 = [
    {'d': d, 'k': 3, 'p': n}
    for d in [3, 4, 5, 6, 8, 10, 12, 15, 18, 21]
    for n in [600, 800, 1000]
]
