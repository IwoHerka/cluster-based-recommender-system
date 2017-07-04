import abc
import argparse
import random
import operator

from collections import defaultdict

from utils import log, calc_averages, sample_unrated_items, load_probe_set


class Recommender(metaclass=abc.ABCMeta):
    def __init__(self):
        # List of integers.
        self.items = None
        
        # Rating data (2D dictionary) of the form:
        # [<int> user][<int> item] = <float> rating.
        self.ratings = None

        # Prediction data. Analogous to [ratings].
        self.predictions = defaultdict(lambda: {})

        # Mapping between user and his
        # leaf cluster: [<int> user] = <cluster> leaf.
        # From leaf, dendrogram is then traversed upwards.
        self.user2cluster = None
         
        # Average rating for each user from [ratings].
        self.avg_ratings = {}
        
        # Global average. Used as fallback for cold-start users.
        self.global_avg_rating = 0

        self.rating_sum = 0
        self.ratings_path = None
        self.min_nraters = None
        self.round_precision = None
        self.iter_limit = None # Need this?
        self.sample_size = None
        self.probe_set_path = None
        

    @abc.abstractmethod
    def predict(self, user, item):
        pass
        
    @abc.abstractmethod    
    def _cluster_users(self):
        pass

    @abc.abstractmethod
    def _com_queue(self, user):
        pass

    @abc.abstractmethod
    def _com2nodes(self, com):
        pass

    def _round(self, n):
        assert self.round_precision
        
        correction = 0.5 if n >= 0 else -0.5
        return int(n / self.round_precision + correction) * self.round_precision

    def _load_ratings(self, path):
        assert path
        assert type(path) == str
        
        ratings = defaultdict(lambda: {})
        items = set([])

        with open(path, 'r') as data:
            for line in data.readlines():
                line = [e for e in line.rstrip().split(self.delimiter)]
                
                user = int(line[0])
                item = int(line[1])
                rating = float(line[2])
                
                items.add(item)
                ratings[user][item] = rating

        #count = 0
        #lratings = {}

        # Relabel keys: 0-N.
        #for key in ratings.keys():            
        #    lratings[count] = ratings[key]
        #    count += 1

        return (ratings, items)

    def _load_tuples(self, path):
        tuples = []

        with open(path, 'r') as data:
            for line in data.readlines():
                line = line.rstrip().split(self.delimiter)
                tuples.append((int(line[0]), int(line[1])))

        return tuples
                

    def prepare(self):
        assert self.ratings_path
        
        log('Loading ratings...')
        self.ratings, self.items = self._load_ratings(self.ratings_path)

        log('Calculating average ratings...')
        self.avg_ratings = calc_averages(self.ratings)
        self.rating_sum = sum(self.avg_ratings.values())
        self.global_avg_rating = self.rating_sum / len(self.ratings)

        log('Clustering users...', 0, 1)
        self.user2cluster = self._cluster_users()
        
    def test(self):
        assert self.ratings
        assert self.sample_size
        assert type(self.probe_set_path) == str

        #print(self.predict(15, 2699))
        #return

        N = 10
        T = 25
        hits = 0
        self.sample_size = 1000
        
        top_ratings = self._load_tuples(self.probe_set_path)
        
        for user, item in top_ratings[:T]:              
            rated_items = list(self.ratings[user]) if user in self.ratings else []
            unrated_items = sample_unrated_items(self.ratings,
                                                 self.sample_size - 1,
                                                 exclude=rated_items)

            sample_set = [item] + unrated_items
            ratings = []
            count = 0

            for it in sample_set:
                count += 1
                #print('iteration', count)
                score = self.predict(user, it)[0]
                if it == item:
                    SCORE = score
                ratings.append((it, score))

            ratings.sort(key=lambda k: k[1], reverse=True)
            ratings = [r[0] for r in ratings[:N]]
            
            log('\n[user]: {}\n'
                '[item]: {}\n'
                '[score]: {}\n'
                '[in top-N]: {}'.format(user, item, SCORE, item in ratings), 1)

            hits += int(item in ratings)

        print(hits, N, self.sample_size, hits / T)


        
#666
#206

#10/100: 0.34
#10/500: 
