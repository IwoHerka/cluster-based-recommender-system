import sys
import time
import argparse

import networkx as nx
import matplotlib.pyplot as plt

# Compiled Infomap.
from infomap import infomap


global silent
shortest_paths = {}
avg_ratings = {}
global_avg_rating = None


def log(msg):
    if not silent:
        sys.stdout.write(msg)

        
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


def preprocess(path, target):
    with open(path, 'r') as src:
        with open(target, 'w') as out:
            for line in src.readlines():
                out.write(line.rsplit(' ', 1)[0] + '\n')
    
    
def load_trust(path):
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
        if node.isLeaf and node.originalLeafIndex != 0:
            cid = ''

            for i in node.path():
                cid += str(i + 1)
                
                if not cid in com2nodes:
                    com2nodes[cid] = set()

                # ID of the last community in a path contains
                # information about previous communities,
                # so we can safely override the current value.
                node2com[node.originalLeafIndex] = cid
                
                com2nodes[cid].add(node.originalLeafIndex)

                cid += ':'

    log('Done.\n\n    [communities in total]: {}\n\n'.format(len(com2nodes)))
    
    return (com2nodes, node2com)


def load_ratings(path, items):
    ratings = {}
    
    with open(path, 'r') as data:
        for line in data.readlines():
            rating = [e for e in line.replace('\n', '').split(' ')]
            items.add(int(rating[1]))

            if not int(rating[0]) in ratings:
                ratings[int(rating[0])] = {}

            ratings[int(rating[0])][int(rating[1])] = float(rating[2])

    return ratings               


def predict(trust_net, trust_coms, com_id, ratings, user, item):
    rating = 0
    len_sum = -1
    weights = []
    unreachable = 0
    com_paths = []
    com_queue = []
    distances = []
    path_lengths = []
    user2rating = None
    paths_from_user = None

    if user in avg_ratings:
        avg_rating = avg_ratings[user]
    else:
        avg_rating = global_avg_rating
    
    while True:
        com_queue.append(com_id)
        com_id = com_id.rsplit(':', 1)[0]
        if not ':' in com_id:
            com_queue.append(com_id)
            break

    for com_id in com_queue:
        user2rating = []
        
        for rater in trust_coms[com_id]:
            if rater == user:
                continue

            if rater in ratings and item in ratings[rater]:
                user2rating.append((rater, ratings[rater][item]))
        
        if user2rating:
            if not com_id in shortest_paths:
                shortest_paths[com_id] = {}

            if not user in shortest_paths[com_id]:
                subnet = trust_net.subgraph(trust_coms[com_id])
                shortest_paths[com_id][user] = \
                        nx.single_source_shortest_path(subnet, user) 
                
            paths_from_user = shortest_paths[com_id][user]

            if len(user2rating) == 1:
                rater = user2rating[0][0]
                if rater not in paths_from_user:
                    unreachable += 1
                    distance = -1
                else:
                    distance = len(paths_from_user[rater]) - 1
                    
                delta = user2rating[0][1] - avg_ratings[rater]
                return (round(avg_rating + delta), 1, unreachable, distance)
            
            break

    for rater_rating in user2rating:
        rater = rater_rating[0]
        
        if rater in paths_from_user:
            path_lengths.append(len(paths_from_user[rater]) - 1)
        else:
            unreachable += 1
            path_lengths.append(1000000)

    len_sum = sum(path_lengths)
    weight_sum = 0
    delta = 0
    
    for i, rater_rating in enumerate(user2rating):
        weights.append(1 - (float(path_lengths[i]) / len_sum))
        if path_lengths[i] != 1000000:
            distances.append(path_lengths[i])
        weight_sum += weights[i]

    for i, rater_rating in enumerate(user2rating):
        rater = rater_rating[0]
        rating = rater_rating[1]
        change = float(weights[i] * (rating - avg_ratings[rater]))
        
        if change != 0:
             change /= weight_sum
             
        delta += change
             
    num_raters = len(user2rating)
    avg_distance = -1
    
    if distances:
        avg_distance = sum(distances) / len(distances)
    
    if delta:    
        return (round(avg_rating + delta), num_raters, unreachable, avg_distance)
    else:
        return (avg_rating, num_raters, unreachable, avg_distance)
    

def run(ratings_path, trust_path):
    ratings = None
    items = set([])
    rating_sum = 0.
    trust_net = None
    com2nodes = None
    node2come = None
    
    log('1] Loading trust network... ')

    trust_net = load_trust(trust_path)

    log('Done.\n\n    [nodes]: {}\n    [edges]: {}\n\n'.format(
        trust_net.number_of_nodes(), trust_net.number_of_edges()))
    log('2] Clustering trust network... \n\n')

    start = time.time()
    tmp = cluster(trust_net)
    com2nodes = tmp[0]
    node2com = tmp[1]
    
    log('    (Clustering finished in: {}.)\n\n'.format(time.time() - start))
    log('3] Loading ratings data... ')

    ratings = load_ratings(ratings_path, items)

    log('Done.\n\n    [raters]: {}\n    [items]: {}\n\n'.format(len(ratings), len(items)))
    log('4] Calculating average ratings... ')

    for user in ratings:
        avg_ratings[user] = float(sum(ratings[user].values())) / len(ratings[user])
        rating_sum += avg_ratings[user]

    #print(avg_ratings)    
    global_avg_rating = rating_sum / len(ratings)
    
    log('Done.\n\n5] Rating items... ')

    count = 0
    error = 0.
    unreachable = 0.
    raters = 0.
    avg_distance = 0.
    histogram = [0, 0, 0, 0, 0]
    
    for rater in ratings:
        for item in ratings[rater]:
            count += 1
            
            res = predict(
                trust_net,
                com2nodes,
                node2com[rater],
                ratings,
                rater,
                item
            )

            prediction = res[0]
            raters += res[1]
            unreachable += res[2]
            avg_distance += res[3]
            
            tmp = abs(ratings[rater][item] - prediction)
            histogram[int(tmp)] += 1
            error += tmp
            

    log('Done in: {}.\n\n'.format(time.time() - start))

    log('     [average error]: {}\n' \
        '     [average # raters]: {}\n' \
        '     [error histogram]: {}\n' \
        '     [average distance]: {}\n' \
        '     [unreachable raters]: {}\n\n'.format(
            error / count,
            raters / count,
            histogram,
            avg_distance / count,
            unreachable / raters
        )
    )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        '''Calculate item rating predictions based on trust network
        and users' rating data. Technique uses user's trusted raters to 
        make a prediction (collaborative-filtering). Raters are determined
        by hierarchical clustering, using Infomap algorithm.'''
    )
    
    parser.add_argument(
        'r',
        metavar='ratings',
        type=str,
        help='path to ratings data, [<user> <item> <rating>] format'
    )

    parser.add_argument(
        't',
        metavar='trust',
        type=str,
        help='path to trust net, NetworkX\'s edge-list format'
    )

    parser.add_argument(
        '-s',
        '--silent',
        type=bool,
        help='do not output any logs'
    )

    args = parser.parse_args()
    silent = args.silent
    run(args.r, args.t)
    

    
