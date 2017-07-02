import sys
from random import sample

from scipy.sparse import dok_matrix


def log(msg, tab=0, bpad=0):
    global silent
    
    if not False:
        msgs = msg.split('\n') if '\n' in msg else [msg]

        for msg in msgs:
            for t in range(tab):
                sys.stdout.write('  ')
            
            sys.stdout.write(msg + '\n')

        if bpad:
            sys.stdout.write('\n')


def calc_averages(dictionary):
    averages = {}

    for key in dictionary:
        values = dictionary[key].values()
        averages[key] = float(sum(values)) / len(values)

    return averages


def sample_unrated_items(ratings, nsamples=1, exclude=[]):
    count = 0
    samples = set([])
    
    if not ratings:
        return samples

    while count < nsamples:
        for _ in range(nsamples):
            user = sample(ratings.keys(), 1)[0]
            item = sample(ratings[user].keys(), 1)[0]

            if not item in exclude:
                exclude.append(item)
                samples.add(item)
                count += 1

                if count >= nsamples:
                    break

    return list(samples)


def load_probe_set(path, div='::'):
    ratings = set([])
    print(div)

    with open(path, 'r') as src:
        for ln in [ln.split(div) for ln in src.readlines()]:
            ratings.add((int(ln[0]), int(ln[1])))

    return ratings

            
            
    
