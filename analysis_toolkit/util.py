"""DataFrame utilities
"""

import numpy as np


def filter(df, **kwargs):
    for k, v in kwargs.items():
        df = df[df[k] == v]
    return df


def oracle_ratio(df, key='rand', **kwargs):

    df = filter(df, **kwargs)
    return list(df[key] / df['oracle_' + key])


def get_data(
        df, test, key,
        sortstat=np.mean, sortkey=None, methods={}):

    data = [
        oracle_ratio(df, method=m, key=key, **test)
        for m in methods.keys()]

    sortlist = [
        sortstat(d) for d in (
            data if sortkey is None else
            [
                oracle_ratio(df, method=m, key=sortkey, **test)
                for m in methods.keys()
            ]
        )]

    labels = [v for k, v in methods.items()]

    return list(zip(*sorted(zip(sortlist, data, labels))))
