
import json

from util import get_files_recursive

EVAL_DIRS = ["./eval"]

COLUMNS = [
    'rand', 'nmi',
    'oracle_rand', 'oracle_nmi',
    'num_clusters', 'best_idx',
    'aggregation', 'segregation',
    'oracle_aggregation', 'oracle_segregation',
    'iterations',
    'method', 'd', 'p', 'k', 'r',
    'id', 'file'
]


def load(file, base):

    fn = file.replace(base + '/', '')

    method, params, test_id = fn.split('/')

    d, p, k, r = [s.split('=')[1] for s in params.split(',')]

    with open(file) as f:
        data = json.loads(f.read())

    data.update({
        "d": d, "p": p, "k": k, "r": r,
        "method": method,
        "id": test_id.replace('.npz_scores.json', ''),
        "file": '"' + file + '"'
    })

    for score in ['rand', 'nmi', 'aggregation', 'segregation']:
        if 'oracle_' + score not in data:
            data['oracle_' + score] = 1

    return data


def process_dir(base, f):

    files = get_files_recursive(base, ext='.json')

    f.write(','.join(COLUMNS) + '\n')
    for fn in files:
        data = load(fn, base)
        f.write(','.join(str(data.get(k, 0)) for k in COLUMNS) + '\n')


if __name__ == '__main__':

    with open('summary.csv', 'w') as f:
        for base in EVAL_DIRS:
            process_dir(base, f)
