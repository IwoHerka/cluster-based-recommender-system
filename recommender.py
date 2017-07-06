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
        self.min_nraters = None
        self.round_precision = None
        
        self.top_ratings_path = None
        self.ratings_path = None

        self.top_n = None
        self.test_sample_size = None
        

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

    def _load_ratings(self, path, relabel=False):
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

        if relabel:
            count = 0
            lratings = {}

            # Relabel keys: 0-N.
            for key in ratings.keys():            
                lratings[count] = ratings[key]
                count += 1

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
        
        log('Loading ratings from {}...'.format(self.ratings_path))
        self.ratings, self.items = self._load_ratings(self.ratings_path)

        log('Calculating average ratings...')
        self.avg_ratings = calc_averages(self.ratings)
        self.rating_sum = sum(self.avg_ratings.values())
        self.global_avg_rating = self.rating_sum / len(self.ratings)

        log('Clustering users...', 0, 1)
        self.user2cluster = self._cluster_users()
        
    def test(self):
        assert self.ratings
        assert type(self.top_ratings_path) == str
        assert type(self.top_n) == int and self.top_n > 0
        assert type(self.test_sample_size) == int and self.test_sample_size > 0

        log('Loading top ratings from {}...'.format(self.top_ratings_path))
        top_ratings = self._load_tuples(self.top_ratings_path)

        hits = 0
        cnt = 0
        
        for user, top_item in top_ratings:            
            rated_items = list(self.ratings.get(user, []))
            
            unrated_items = sample_unrated_items(
                self.ratings,
                self.test_sample_size - 1,
                exclude=rated_items
            )

            ratings = []
            sample_items = [top_item] + unrated_items

            for item in sample_items:
                ratings.append((item, self.predict(user, item)))

            ratings.sort(key=lambda k: k[1], reverse=True)
            top_N = [r[0] for r in ratings[:self.top_n]]
            
            if not self.silent:
                log('\n'
                    '[user]: {}\n'
                    '[item]: {}\n'
                    '[top-N]: {}\n'
                    '[score]: {}'
                    .format(user, top_item, top_item in top_N,
                            self.predict(user, top_item)), 1)

            if self.predict(user, top_item) == -1:
                cnt += 1
            hits += int(top_item in top_N)

        log('Test results:', 0, 1)
        log('[N]: {}\n'
            '[hits]: {}\n'
            '[test sample]: {}\n'
            '[iterations]: {}\n'
            '[recall]: {}'.format(hits,
                                  self.top_n,
                                  self.test_sample_size,
                                  len(top_ratings),
                                  hits / len(top_ratings)), 1)
        print(cnt)

    def _get_parser(self):
        parser = argparse.ArgumentParser(
            description='''[?]'''
        )

        parser.add_argument(
            'ratings_path',
            type=str,
            help='Path to ratings data, [<user> <item> <rating>] format.'
        )

        parser.add_argument(
            'top_ratings_path',
            type=str,
            help='Path to top ratings used in testing stage. '
            'Same format as ratings_path.'
        )

        parser.add_argument(
            'round_precision',
            type=int,
            help='Prediction rounding precision. '
            '1 produces integers, 0.5 produces 0, 0.5, 1, 1.5, etc.'
        )

        parser.add_argument(
            'top_n',
            type=int,
            help='N, number of top items to calculate recall.'
        )

        parser.add_argument(
            '-min-raters',
            type=int,
            default=1,
            help='Minimal acceptable number of raters used to make a prediction. '
            'Value of k guarantees *at least* k raters. Defaults to 1.'
        )

        parser.add_argument(
            '-linkage-method',
            type=str,
            default='average',
            help='Clustering linkage method. Default: average.'
        )

        parser.add_argument(
            '-linkage-metric',
            type=str,
            default='cosine',
            help='Clustering linkage metric. Default: cosine.'
        )

        parser.add_argument(
            '-test-sample-size',
            type=int,
            default=1000,
            help='Test sample size. Default: 1000.'
        )

        parser.add_argument(
            '-delimiter',
            type=str,
            default=' ',
            help='Source data delimiter. Default is one whitespace.'
        )

        parser.add_argument(
            '-silent',
            type=bool,
            default=False,
            help='Turn verbosity on/off.'
        )

        return parser

    def init(self, args):
        self.ratings_path = args.ratings_path
        self.top_ratings_path = args.top_ratings_path
        self.round_precision = args.round_precision
        self.top_n = args.top_n
    
        self.min_nraters = args.min_raters
        self.test_sample_size = args.test_sample_size
        self.linkage_method = args.linkage_method
        self.linkage_metric = args.linkage_metric
        self.delimiter = args.delimiter
        self.silent = args.silent
