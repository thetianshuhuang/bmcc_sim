
import json

from util import get_files_recursive

EVAL_DIR = "./eval"

COLUMNS = [
    'rand', 'nmi',
    'oracle_rand', 'oracle_nmi',
    'num_clusters', 'best_idx',
    'aggregation', 'segregation',
    'oracle_aggregation', 'oracle_segregation',
    'iterations',
    'method', 'd', 'p', 'k', 'r',
    'id'
]


def load(file):

    fn = file.replace(EVAL_DIR + '/', '')

    method, params, test_id = fn.split('/')

    d, p, k, r = [s.split('=')[1] for s in params.split(',')]

    with open(file) as f:
        data = json.loads(f.read())

    data.update({
        "d": d, "p": p, "k": k, "r": r,
        "method": method,
        "id": test_id.replace('.npz_scores.json', '')})

    for score in ['rand', 'nmi', 'aggregation', 'segregation']:
        if score not in data:
            data[score] = 1

    return data


if __name__ == '__main__':

    files = get_files_recursive(EVAL_DIR, ext='.json')

    with open('summary.csv', 'w') as f:
        f.write(','.join(COLUMNS) + '\n')
        for fn in files:
            data = load(fn)
            f.write(','.join(str(data.get(k, 0)) for k in COLUMNS) + '\n')
