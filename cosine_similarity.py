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


global silent
silent = False

cluster = KMeans

log('Loading rating data...', 0)

pack = load_ratings('./data/filmtrust_ratings.dat')
ratings = pack[0]
items = pack[1]

nrows = max(items) + 1
ncols = max(ratings) + 1

log('Bulding sparse matrix...', 0, 1)

R = dok_matrix((nrows, ncols), dtype=np.float32)

for user in ratings:
    for item in ratings[user]:
        R[item, user] = ratings[user][item]

log('[rows - items]: {}'.format(nrows), 1)
log('[columns - users]: {}'.format(ncols), 1)        
log('[stored values]: {}'.format(R.nnz), 1, 1)
log('Calculating cosine similarity...', 0)
log('Clustering...', 0, 1)


#X = R.toarray()


#X = np.concatenate([np.random.randn(3, 10), np.random.randn(2, 10) + 100])
#model = FeatureAgglomeration()
#model.fit(X)

#ii = itertools.count(X.shape[0])
#x = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]

#print(model.labels_)
#for l in model.labels_:
#    print(l)
#print(x)



#cossim = cosine_similarity(R.transpose())
#labels = cluster(2).fit_predict(cossim)

R = R.transpose().toarray()

print(R)

Z = hierarchy.linkage(R, method='single', metric='cosine')
plt.figure()
dn = hierarchy.dendrogram(Z)
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z, ax=axes[1], above_threshold_color='#bcbddc',
                           orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
plt.show()

