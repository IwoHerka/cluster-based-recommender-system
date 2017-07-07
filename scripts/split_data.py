import sys
import random
import fileinput
import random
import time
from random import sample
from collections import defaultdict

DIV = ' '
SAMPLE_SIZE = 500
RATINGS = './trust_data/ratings.txt'

TRAINING = './trust_data/training.dat'
TOP = './trust_data/top.dat'


def load_ratings(path):
    assert path
    assert type(path) == str

    ratings = defaultdict(lambda: {})

    with open(path, 'r') as data:
        for line in data.readlines():
            line = [e for e in line.rstrip().split(DIV)]

            user = int(line[0])
            item = int(line[1])
            rating = float(line[2])

            ratings[user][item] = rating

    #count = 0
    #lratings = {}

    # Relabel keys: 0-N.
    #for key in ratings.keys():            
    #    lratings[count] = ratings[key]
    #    count += 1

    return ratings


def sample_unrated_items(ratings, nsamples=1, exclude=[]):
    count = 0
    samples = []
    
    if not ratings:
        return samples

    while count < nsamples:
        user = sample(ratings.keys(), 1)[0]
            
        if not ratings[user]:
                del ratings[user]
                continue

        random.seed(time.time())
            
        item = sample(ratings[user].keys(), 1)[0]

        if len(ratings[user]) > 1:
            samples.append((user, item, ratings[user][item]))
            del ratings[user][item]
            count += 1
            print(count)

            if count >= nsamples:
                break
                
    return samples


count = 0
samples = []
ratings = load_ratings(RATINGS)
probe_set = sorted(sample_unrated_items(ratings, SAMPLE_SIZE, []))
       

with open(TOP, 'w') as top:
    for user, item, rating in probe_set:
        if rating == 4:
            top.write('{} {} {}\n'.format(user, item, rating))


with open(TRAINING, 'w') as training:
    for user in ratings:
        for item in ratings[user]:
            training.write('{} {} {}\n'.format(user, item, ratings[user][item]))
