import sys
import time

import networkx as nx
import matplotlib.pyplot as plt

# Compiled Infomap.
from infomap import infomap


log = sys.stdout.write


def draw(G, partition):
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.

    for com in set(partition.values()):
        count += 1
        list_nodes = [v for v in partition.keys() if partition[v] == com]
        nx.draw_networkx_nodes(G,
                               pos,
                               list_nodes,
                               node_size=20,
                               node_color=plt.cm.jet(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    
def load_trust(path):
    return nx.karate_club_graph()
#return nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)


def cluster(G):
    infomap_wrapper = infomap.Infomap('-d --tree --silent')

    log('    1] Building Infomap network from a NetworkX graph... ')

    for e in G.edges_iter():
        infomap_wrapper.addLink(*e)

    log('Done.\n    2] Finding communities with Infomap... ')
    
    com2nodes = {}
    node2com = {}
    infomap_wrapper.run()
    tree = infomap_wrapper.tree
    
    for node in tree.treeIter():
        if node.isLeaf:
            cid = ''

            for i in node.path():
                cid += str(i + 1)
                
                if not cid in com2nodes:
                    com2nodes[cid] = set()

                # ID of the last community in a path contains
                # information about previous communities,
                # so we can safely override the current value.
                node2com[node.originalLeafIndex + 1] = cid
                
                com2nodes[cid].add(node.originalLeafIndex + 1)

                cid += ':'

    log('Done.\n\n    [communities in total]: {}\n\n'.format(len(com2nodes)))
    
    return (com2nodes, node2com)


def load_ratings(path, items):
    ratings = {}
    
    with open(path, 'r') as data:
        for line in data.readlines():
            rating = [int(e.replace('\n', '')) for e in line.split(' ')]
            items.add(rating[1])

            if not rating[0] in ratings:
                ratings[rating[0]] = {}

            ratings[rating[0]][rating[1]] = rating[2]

    return ratings               


def predict(user, item, ratings, trust_net, community):
    rating = 0

    log('        Checking neighbors for (u:{})...\n'.format(user))
    
    #if nbor in ratings and item in ratings[nbor]:
    #nrating = ratings[nbor][item]
    #log('        Found the rating in neighbor set: (u:{}, r:{})\n'.format(nbor, nrating))


if __name__ == '__main__':
    TEST_DIR = './test_data/'
    TRUST_DAT = 'trust.dat'
    RATING_DAT = 'ratings.dat'

    log('1] Loading trust network... ')

    G = load_trust('../Matlab/Networks/Gnutella.dat')#TEST_DIR + TRUST_DAT)
    nnodes = G.number_of_nodes()
    nedges = G.number_of_edges()

    log('Done.\n\n    [nodes]: {}\n    [edges]: {}\n\n'.format(nnodes, nedges))
    log('2] Clustering trust network... \n\n')

    start = time.time()
    res = cluster(G)
    com2nodes = res[0]
    node2com = res[1]
    
    log('    (Clustering finished in: {}.)\n\n'.format(time.time() - start))
    log('3] Loading ratings data... ')

    items = set([])
    ratings = load_ratings(TEST_DIR + RATING_DAT, items)

    log('Done.\n\n    [raters]: {}\n    [items]: {}\n\n'.format(len(ratings), len(items)))
    log('4] Rating items... \n\n')

    users = [1]
    for user in users:
        for item in list(items)[:10]:
            log('    Predicting for (u:{}, i:{})... \n\n'.format(user, item))
            
            predict(user, item, ratings, G, node2com[user])

    log('Computing shortest paths... ')

    
        

    log('Done in: {}.\n\n'.format(time.time() - start))    
