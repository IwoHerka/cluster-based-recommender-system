import argparse

import numpy as np
from recommender import Recommender
from scipy.sparse import dok_matrix
from scipy.cluster import hierarchy

from utils import log


class cluster:
    def __init__(self, id=-1):
        self.id = id
        self.parent = None
        self.children = set([])

    @property
    def nodes(self, except_for=None):
        nodes = set()

        if self.children:
            for child in self.children:
                if except_for == child:
                    continue

                nodes = nodes.union(child.nodes)
            return nodes
        
        return set([self.id])

    def __repr__(self):
        return str([c.id for c in self.children])

    
class SimilarityRecommender(Recommender):       
    def _cluster_users(self):
        assert self.ratings
        assert self.items

        clusters = {}
        nrows, ncols = len(self.ratings), max(self.items) + 1
        dokmat = dok_matrix((nrows, ncols), dtype=np.float32)

        for user in self.ratings:
            for item in self.ratings[user]:
                dokmat[user, item] = self.ratings[user][item]

        # TODO: Find a way to use sparse matrix.
        linkage = hierarchy.linkage(dokmat.toarray(),
                                    method='complete',
                                    metric='cosine')
        
        for ind, row in enumerate(linkage):
            for i in range(2):
                rid = int(row[i])
                if not rid in clusters:
                    clusters[rid] = cluster(rid)

            A = clusters.get(int(row[0]))
            B = clusters.get(int(row[1]))
            C = cluster(nrows + ind)
            
            C.children.add(A)
            C.children.add(B)
            A.parent = C
            B.parent = C

            clusters[C.id] = C

        log('[rows - users]: {}\n'
            '[columns - items]: {}\n'      
            '[stored values]: {}\n'
            '[clusters]: {}'
            .format(nrows, ncols, dokmat.nnz, len(clusters)), 1, 1)

        #print(dokmat.toarray())
        #print('')
        #print(clusters)
        
        return clusters

    def _com_queue(self, user):
        assert self.user2cluster
        com = self.user2cluster[user]
        queue = [com]

        while com.parent:
            queue.append(com.parent)
            com = com.parent

        return queue

    def _com2nodes(self, com):
        return com.nodes
        
    def _predict(self, user, item):
        assert self.ratings
        assert self.min_nraters
        assert self.avg_ratings
        
        com_queue = self._com_queue(user)
        
        for com in com_queue:
            raters = []
            
            for rater in [v for v in self._com2nodes(com) if v != user]:
                if rater in self.ratings and item in self.ratings[rater]:
                    raters.append((rater, self.ratings[rater][item]))

            if len(raters) >= self.min_nraters:
                break
        
        delta = 0
        #print(raters)
        
        for rater, rating in raters:
            delta += rating - self.avg_ratings[rater]

        average = max(self.avg_ratings.get(user, -1), self.global_avg_rating)
        prediction = self._round(average + delta)
        
        return (prediction, len(raters), 0, 0)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''[?]'''
    )
    
    parser.add_argument(
        'r',
        type=str,
        help='Path to ratings data, [<user> <item> <rating>] format.'
    )

    parser.add_argument(
        '-min-raters',
        type=int,
        default=1,
        help='Minimal acceptable number of raters used to make a prediction. '
             'Value of k guarantees *at least* k raters. Defaults to 1.'
    )

    args = parser.parse_args()
    recommender = SimilarityRecommender()
    recommender.ratings_path = args.r
    recommender.min_nraters = args.min_raters
    recommender.round_precision = 0.5
    recommender.run()
