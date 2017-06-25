import sys
import networkx as nx
import scipy
import argparse
from scipy.sparse import dok_matrix
from scipy.cluster import hierarchy
import numpy as np


global silent

def log(msg, tab=0, bpad=0):
    global silent
    
    if not silent:
        msgs = msg.split('\n') if '\n' in msg else [msg]

        for msg in msgs:
            for t in range(tab):
                sys.stdout.write('  ')
            
            sys.stdout.write(msg + '\n')

        if bpad:
            sys.stdout.write('\n')

            
def relabel(dictionary):
    count = 0
    ret = {}

    for key in dictionary.keys():            
        ret[count] = dictionary[key]
        count += 1

    return ret


def load_ratings(path):
    ratings = {}
    items = set([])

    with open(path, 'r') as data:
        for line in data.readlines():
            rating = [e for e in line.replace('\n', '').split(' ')]
            items.add(int(rating[1]))

            if not int(rating[0]) in ratings:
                ratings[int(rating[0])] = {}

            ratings[int(rating[0])][int(rating[1])] = float(rating[2])

    return (relabel(ratings), items)


def to_matrix(ratings, nrows, ncols):
    R = dok_matrix((nrows, ncols), dtype=np.float32)

    for user in ratings:
        for item in ratings[user]:
            R[user, item] = ratings[user][item]

    return R

   
class SimilarityRecommender:
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

    def run(self, ratings_path):
        log('Loading rating data...', 0)
        
        ratings, items = load_ratings(ratings_path)
        nrows, ncols = max(ratings) + 1, max(items) + 1

        log('Bulding sparse matrix...', 0, 1)
        
        dok_mat = to_matrix(ratings, nrows, ncols)

        log('[rows - items]: {}\n'
            '[columns - users]: {}\n'      
            '[stored values]: {}'
            .format(nrows, ncols, dok_mat.nnz), 1, 1)
        
        log('Clustering...', 0, 1)
        
        Z = hierarchy.linkage(dok_mat.toarray(), method='ward')
        clusters = {}

        for ind, row in enumerate(Z):
            for i in range(2):
                if not row[i] in clusters:
                    clusters[row[i]] = self.cluster(row[i])

            A = clusters.get(row[0])
            B = clusters.get(row[1])
            C = self.cluster(nrows + ind)
            
            C.children.add(A)
            C.children.add(B)
            A.parent = C
            B.parent = C

            clusters[C.id] = C

        log('[clusters]: {}'.format(len(clusters)), 1, 1)

        


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

    parser.add_argument(
        '-s',
        '--silent',
        type=bool,
        help='do not output any logs'
    )

    args = parser.parse_args()
    silent = args.silent
    recommender = SimilarityRecommender()
    recommender.run(args.r)
    
