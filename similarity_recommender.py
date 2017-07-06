import argparse

import numpy as np
from recommender import Recommender
from scipy.sparse import dok_matrix
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

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
    def __init__(self):
        super(SimilarityRecommender, self).__init__()

        # https://docs.scipy.org/doc/scipy-0.19.0/reference/ \
        # generated/scipy.cluster.hierarchy.linkage.html
        self.linkage_method = None
        self.linkage_metric = None
        
    def _cluster_users(self):
        assert self.linkage_method
        assert self.linkage_metric
        assert self.ratings
        assert self.items

        clusters = {}
        nrows, ncols = len(self.ratings), max(self.items) + 1
        self.dokmat = dok_matrix((nrows, ncols), dtype=np.float32)

        for user in self.ratings:
            for item in self.ratings[user]:
                self.dokmat[user, item] = self.ratings[user][item]

        # TODO: Find a way to use sparse matrix.
        linkage = hierarchy.linkage(self.dokmat.toarray(),
                                    method=self.linkage_method,
                                    metric=self.linkage_metric)
        
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
            .format(nrows, ncols, self.dokmat.nnz, len(clusters)), 1, 1)
        
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
    
    def predict(self, user, item):
        assert self.ratings
        assert self.min_nraters
        assert self.avg_ratings
        
        com_queue = self._com_queue(user)
        
        for com in com_queue:
            raters = []
            
            for rater in [v for v in self._com2nodes(com) if v != user]:
                if rater in self.ratings and item in self.ratings[rater]:
                    raters.append((rater, self.ratings[rater][item]))

            if raters and len(raters) >= self.min_nraters:
                break

        average = self.avg_ratings.get(user, self.global_avg_rating)
        
        if not raters:
            return average
        
        delta = 0.      
        weight_sum = len(raters)
        
        for i, (rater, rating) in enumerate(raters):
            delta += rating - self.avg_ratings[rater]

        wdelta = delta / weight_sum if weight_sum != 0 else delta

        return min(5, max(0, self._round(average + wdelta)))
    

if __name__ == '__main__':
    recommender = SimilarityRecommender()
    parser = recommender._get_parser()
    args = parser.parse_args()
    
    recommender.init(args)
    recommender.prepare()
    recommender.test()
