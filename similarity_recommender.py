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
        return self.id

    
class SimilarityRecommender(Recommender):       
    def _cluster_users(self):
        assert self.ratings
        assert self.items

        clusters = {}
        nrows, ncols = max(self.ratings) + 1, max(self.items) + 1
        dokmat = dok_matrix((nrows, ncols), dtype=np.float32)

        for user in self.ratings:
            for item in self.ratings[user]:
                dokmat[user, item] = self.ratings[user][item]

        # TODO: Find a way to use sparse matrix.
        linkage = hierarchy.linkage(dokmat.toarray(), method='ward')
        
        for ind, row in enumerate(linkage):
            for i in range(2):
                if not row[i] in clusters:
                    clusters[row[i]] = cluster(row[i])

            A = clusters.get(row[0])
            B = clusters.get(row[1])
            C = cluster(nrows + ind)
            
            C.children.add(A)
            C.children.add(B)
            A.parent = C
            B.parent = C

            clusters[C.id] = C

        log('[rows - items]: {}\n'
            '[columns - users]: {}\n'      
            '[stored values]: {}\n'
            '[clusters]: {}'
            .format(nrows, ncols, dokmat.nnz, len(clusters)), 1, 1)
        
        return clusters

    def _predict(self, user, item):
        return (1, 1, 1, 1)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''[?]'''
    )
    
    parser.add_argument(
        'r',
        metavar='ratings',
        type=str,
        help='path to ratings data, [<user> <item> <rating>] format'
    )

    args = parser.parse_args()
    recommender = SimilarityRecommender()
    recommender.ratings_path = args.r
    recommender.run()
