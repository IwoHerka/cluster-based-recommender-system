import abc
from collections import defaultdict

from utils import log, calc_averages


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
        
    @abc.abstractmethod    
    def _cluster_users(self):
        pass

    @abc.abstractmethod
    def _predict(self, user, item):
        pass

    @abc.abstractmethod
    def _com_queue(self, user):
        pass

    @abc.abstractmethod
    def _com2nodes(self, com):
        pass

    def _load_ratings(self, path):
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
        
    def run(self):
        assert self.ratings_path
        
        log('Loading ratings...')
        self.ratings, self.items = self._load_ratings(self.ratings_path)

        log('Calculating average ratings...')
        self.avg_ratings = calc_averages(self.ratings)
        self.rating_sum = sum(self.avg_ratings.values())
        self.global_avg_rating = self.rating_sum / len(self.ratings)

        log('Clustering users...', 0, 1)
        self.user2cluster = self._cluster_users()

        iters = 0
        avg_error = 0
        self.predictions = defaultdict(lambda: {})
        
        #for rater in self.ratings:
        for rater in [0]:    
            ratings = self.ratings[rater]

            for item in [1]:
            #for item in ratings:
                prediction, nraters, ndisc, avg_dist = self._predict(rater, item)
                error = abs(ratings[item] - prediction)
                self.predictions[rater][item] = prediction
                
