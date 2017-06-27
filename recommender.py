import abc
import argparse
import random
import operator

from collections import defaultdict

from utils import log, calc_averages, sample_ratings, load_items


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
                line = [e for e in line.rstrip().split(' ')]
                
                user = int(line[0])
                item = int(line[1])
                rating = float(line[2])
                
                items.add(item)
                ratings[user][item] = rating

        count = 0
        lratings = {}

        # Relabel keys: 0-N.
        for key in ratings.keys():            
            lratings[count] = ratings[key]
            count += 1

        return (lratings, items)

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
            
        user = random.sample(self.ratings.keys(), 1)[0]
        rated_items = list(self.ratings[user])
        top_items = load_items(self.probe_set_path)
        sample_set = sample_ratings(self.ratings, self.sample_size, exclude=rated_items)

        
