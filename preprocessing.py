import sys
import time

import networkx as nx
import matplotlib.pyplot as plt

# Compiled Infomap.
from infomap import infomap


log = sys.stdout.write
shortest_paths = {}
avg_ratings = {}
global_avg_rating = None


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
    #return nx.karate_club_graph()
    return nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)


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


def predict(trust_net, trust_coms, com_id, ratings, user, item):
    rating = 0
    len_sum = -1
    weights = []
    com_paths = []
    com_queue = []
    path_lengths = []
    user2rating = None
    paths_from_user = None
    
    while True:
        com_queue.append(com_id)
        com_id = com_id.rsplit(':', 1)[0]
        if not ':' in com_id:
            com_queue.append(com_id)
            break

    log('        Looking for raters... \n\n')    

    for com_id in com_queue:
        user2rating = []
        
        for rater in trust_coms[com_id]:
            if rater in ratings and item in ratings[rater]:
                user2rating.append((rater, ratings[rater][item]))

        log('        [raters in {}]: {}\n'.format(com_id, len(user2rating)))
        
        if user2rating:
            if len(user2rating) == 1:
                return user2rating[0][1]
                
            if not com_id in shortest_paths:
                shortest_paths[com_id] = {}

            if not user in shortest_paths[com_id]:
                subnet = nx.Graph(trust_net.subgraph(trust_coms[com_id]))
                shortest_paths[com_id][user] = \
                        nx.single_source_shortest_path(subnet, user) 
                
            paths_from_user = shortest_paths[com_id][user]
            break
   
    for rater_rating in user2rating:
        rater = rater_rating[0]
        
        if rater in paths_from_user:
            path_lengths.append(len(paths_from_user[rater]) - 1)
        else:
            path_lengths.append(1000000)

    len_sum = sum(path_lengths)
    weight_sum = 0
    delta = 0
    
    for i, rater_rating in enumerate(user2rating):
        weights.append(1 - float(path_lengths[i]) / len_sum)
        weight_sum += weights[i]

    for i, rater_rating in enumerate(user2rating):
        rater = rater_rating[0]
        rating = rater_rating[1]
        delta += float(weights[i] * (avg_ratings[rater] - rating)) / weight_sum


    if user in avg_ratings:
        avg_rating = avg_ratings[user]
    else:
        avg_rating = global_avg_rating
        
    return avg_rating + (delta / len(user2rating))
            

if __name__ == '__main__':
    TEST_DIR = './test_data/'
    TRUST_DAT = 'trust.dat'
    RATING_DAT = 'ratings.dat'

    ratings = None
    items = set([])
    rating_sum = 0.
    trust_net = None
    com2nodes = None
    node2come = None
    
    log('1] Loading trust network... ')

    trust_net = load_trust('../Matlab/Networks/Gnutella.dat')#TEST_DIR + TRUST_DAT)

    log('Done.\n\n    [nodes]: {}\n    [edges]: {}\n\n'.format(
        trust_net.number_of_nodes(), trust_net.number_of_edges()))
    log('2] Clustering trust network... \n\n')

    start = time.time()
    tmp = cluster(trust_net)
    com2nodes = tmp[0]
    node2com = tmp[1]
    
    log('    (Clustering finished in: {}.)\n\n'.format(time.time() - start))
    log('3] Loading ratings data... ')

    ratings = load_ratings(TEST_DIR + RATING_DAT, items)

    log('Done.\n\n    [raters]: {}\n    [items]: {}\n\n'.format(len(ratings), len(items)))
    log('4] Calculating average ratings... ')

    
    for user in ratings:
        avg_ratings[user] = float(sum(ratings[user].values())) / len(ratings[user])
        rating_sum += avg_ratings[user]

    global_avg_rating = rating_sum / len(ratings)

    log('Done.\n\n5] Rating items... \n\n')

    users = [i + 1 for i in range(50)]
    for user in users:
        for item in [101]:#list(items)[:10]:
            log('    Predicting for (u:{}, i:{})... \n\n'.format(user, item))
            
            rating = predict(trust_net, com2nodes, node2com[user], ratings, user, item)

            log('\n    Prediction: {}\n\n'.format(rating))

    log('Computing shortest paths... ')
    log('Done in: {}.\n\n'.format(time.time() - start))    
