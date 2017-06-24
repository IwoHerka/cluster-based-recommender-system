import sys
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
import itertools


def log(msg, tab=0, bpad=0):
    global silent
    
    if not silent:
        for t in range(tab):
            sys.stdout.write('  ')
            
        sys.stdout.write(msg + '\n')

        if bpad:
            sys.stdout.write('\n')
            

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

    return (ratings, items)


def relabel(dictionary):
    count = 0
    ret = {}
    
    for key in dictionary.keys():            
        ret[count] = dictionary[key]
        count += 1

    return ret


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

global silent
silent = False

log('Loading rating data...', 0)

pack = load_ratings('./data/filmtrust_ratings.dat')
ratings = pack[0]
items = pack[1]

nrows = max(ratings)
ncols = max(items) + 1

log('Bulding sparse matrix...', 0, 1)

R = dok_matrix((nrows, ncols), dtype=np.float32)

ratings = relabel(ratings)

for user in ratings:
    for item in ratings[user]:
        R[user, item] = ratings[user][item]

log('[rows - items]: {}'.format(nrows), 1)
log('[columns - users]: {}'.format(ncols), 1)        
log('[stored values]: {}'.format(R.nnz), 1, 1)
log('Clustering...', 0, 1)

#R = np.array([[0.1,   2.5],
#              [1.5,   .4 ],
#              [0.3,   1  ],
#              [1  ,   .8 ],
#              [0.5,   0  ],
#              [0  ,   0.5],
#              [0.5,   0.5],
#              [2.7,   2  ],
#              [2.2,   3.1],
#              [3  ,   2  ],
#              [3.2,   1.3]])

Z = hierarchy.linkage(R.toarray(), method='ward')

clusters = {}

for ind, row in enumerate(Z):
    for i in range(2):
        if not row[i] in clusters:
            clusters[row[i]] = cluster(row[i])
           
    A = clusters.get(row[0])
    B = clusters.get(row[1])
    
    C = cluster(nrows + ind)
    C.children.add(A)
    C.children.add(B)

    clusters[C.id] = C

print(len(clusters))
        

#plt.figure()
#dn = hierarchy.dendrogram(Z)
#hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
#fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
#                           orientation='top')
#dn2 = hierarchy.dendrogram(Z, ax=axes[1], above_threshold_color='#bcbddc',
#                           orientation='right')
#hierarchy.set_link_color_palette(None)  # reset to default after use
#plt.show()

