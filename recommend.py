import sys
import time
import argparse

import networkx as nx
import matplotlib.pyplot as plt

# Compiled Infomap.
from infomap import infomap


global silent

def log(msg):
    if not silent:
        sys.stdout.write(msg)

def preprocess(path, target):
    with open(path, 'r') as src:
        with open(target, 'w') as out:
            for line in src.readlines():
                out.write(line.rsplit(' ', 1)[0] + '\n')

class Recommender:
    def __init__(self):
        self.round_precision = 1.0
        self.shortest_paths = {}
        self.avg_ratings = {}
        self.global_avg_rating = 0
        self.ratings = None
        self.items = set([])
        self.rating_sum = 0.
        self.trust_net = None
        self.com2nodes = None
        self.node2come = None
        self.best = []

    def round(self, n):
        assert self.round_precision
        
        correction = 0.5 if n >= 0 else -0.5
        return int(n / self.round_precision + correction) * self.round_precision

    
    def load_trust(self, path):
        self.trust_net = nx.read_edgelist(path,
                                          create_using=nx.DiGraph(),
                                          nodetype=int)

        
    def user_clusters(self, user):
        assert self.node2com
        assert user in self.node2com
        
        com_id = self.node2com[user]
        com_queue = []
        
        while True:
            com_queue.append(com_id)
            com_id = com_id.rsplit(':', 1)[0]
            
            if not ':' in com_id:
                com_queue.append(com_id)
                break

        return com_queue

    
    def cluster(self):
        assert infomap
        assert self.trust_net
        
        self.node2com = {}
        self.com2nodes = {}
        
        infomap_wrapper = infomap.Infomap('-d --tree --silent')

        log('    1] Building Infomap network from a NetworkX graph... ')

        for e in self.trust_net.edges_iter():
            infomap_wrapper.addLink(*e)

        log('Done.\n    2] Finding communities with Infomap... ')

        infomap_wrapper.run()
        tree = infomap_wrapper.tree

        for node in tree.treeIter():
            if node.isLeaf and node.originalLeafIndex != 0:
                cid = ''

                for i in node.path():
                    cid += str(i + 1)

                    if not cid in self.com2nodes:
                        self.com2nodes[cid] = set()

                    # ID of the last community in a path contains
                    # information about previous communities,
                    # so we can safely override the current value.
                    self.node2com[node.originalLeafIndex] = cid
                    self.com2nodes[cid].add(node.originalLeafIndex)
                    cid += ':'

        log('Done.\n\n    [communities in total]: {}\n\n'.format(
            len(self.com2nodes)))


    def load_ratings(self, path):
        self.ratings = {}
        self.items = set([])

        with open(path, 'r') as data:
            for line in data.readlines():
                rating = [e for e in line.replace('\n', '').split(' ')]
                self.items.add(int(rating[1]))

                if not int(rating[0]) in self.ratings:
                    self.ratings[int(rating[0])] = {}

                self.ratings[int(rating[0])][int(rating[1])] = float(rating[2])

                
    def predict(self, user, item):
        weights = []
        weight_sum = 0
        delta = 0
        unreachable = 0
        distances = []
        path_lengths = []
        user2rating = None
        paths_from_user = None
        com_queue = self.user_clusters(user)
        avg_rating = self.avg_ratings.get(user, self.global_avg_rating)

        for com_id in com_queue:
            user2rating = []

            for rater in (r for r in self.com2nodes[com_id] if r != user):
                if rater in self.ratings and item in self.ratings[rater]:
                    user2rating.append((rater, self.ratings[rater][item]))

            if user2rating:
                if not com_id in self.shortest_paths:
                    self.shortest_paths[com_id] = {}

                if not user in self.shortest_paths[com_id]:
                    subnet = self.trust_net.subgraph(self.com2nodes[com_id])
                    self.shortest_paths[com_id][user] = \
                            nx.single_source_shortest_path(subnet, user) 

                paths_from_user = self.shortest_paths[com_id][user]
                break

        if user2rating:
            assert paths_from_user
            
            if len(user2rating) == 1:
                rater = user2rating[0][0]

                if rater not in paths_from_user:
                    unreachable = 1
                    distance = -1
                else:
                    distance = len(paths_from_user[rater]) - 1

                delta = user2rating[0][1] - self.avg_ratings[rater]
                return (self.round(avg_rating + delta / 2), 1, unreachable, distance)

            for rater_rating in user2rating:
                rater = rater_rating[0]

                if rater in paths_from_user:
                    path_lengths.append(len(paths_from_user[rater]) - 1)
                else:
                    unreachable += 1
                    path_lengths.append(1000000)

            total_path_length = sum(path_lengths)
            
            for i, rater_rating in enumerate(user2rating):
                weights.append(1 - (float(path_lengths[i]) / total_path_length))

                if path_lengths[i] != 1000000:
                    distances.append(path_lengths[i])
                    
                weight_sum += weights[i]

            for i, rater_rating in enumerate(user2rating):
                rater = rater_rating[0]
                rating = rater_rating[1]
                change = float(weights[i] * (rating - self.avg_ratings[rater]))
                change = change / weight_sum if change else 0
                delta += change

        num_raters = len(user2rating)
        avg_distance = sum(distances) / len(distances) if distances else -1
        return (self.round(avg_rating + delta), num_raters, unreachable, avg_distance)

    
    def run(self, ratings_path, trust_path):
        log('1] Loading trust network... ')

        self.load_trust(trust_path)

        log('Done.\n\n    [nodes]: {}\n    [edges]: {}\n\n'.format(
            self.trust_net.number_of_nodes(),
            self.trust_net.number_of_edges()))
        log('2] Clustering trust network... \n\n')

        start = time.time()
        self.cluster()

        log('    (Clustering finished in: {}.)\n\n'.format(time.time() - start))
        log('3] Loading ratings data... ')

        self.load_ratings(ratings_path)

        log('Done.\n\n    [raters]: {}\n    [items]: {}\n\n'.format(len(self.ratings), len(self.items)))
        log('4] Calculating average ratings... ')

        self.avg_ratings = {}
        
        for user in self.ratings:
            self.avg_ratings[user] = \
                float(sum(self.ratings[user].values())) / len(self.ratings[user])
            self.rating_sum += self.avg_ratings[user]

        global_avg_rating = self.rating_sum / len(self.ratings)

        log('Done.\n\n5] Rating items... ')

        count = 0
        error = 0.
        unreachable = 0.
        raters = 0.
        avg_distance = 0.
        histogram = [0, 0, 0, 0, 0]

        for rater in self.ratings:
            for item in self.ratings[rater]:
                count += 1

                res = self.predict(rater, item)

                prediction = res[0]
                raters += res[1]
                unreachable += res[2]
                avg_distance += res[3]

                tmp = abs(self.ratings[rater][item] - prediction)
                print(tmp)
                histogram[int(tmp)] += 1
                error += tmp

                #if prediction in [5.0, 4.5]:
                #    print(ratings[rater][item], prediction, tmp)
                #    best.append((rater, item))


        log('Done in: {}.\n\n'.format(time.time() - start))

        log('     [average error]: {}\n' \
            '     [average # raters]: {}\n' \
            '     [error histogram]: {}\n' \
            '     [average distance]: {}\n' \
            '     [average global rating]: {}\n' \
            '     [unreachable raters]: {}\n\n'.format(
                error / count,
                raters / count,
                histogram,
                avg_distance / count,
                global_avg_rating,
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

    parser.add_argument(
        '-p',
        '--precision',
        type=bool,
        help='prediction rounding precision, e.g. precision of 0.5 will ' \
        'produce ratings: 0, 0.5, 1, 1.5, ...'
    )

    args = parser.parse_args()
    recommender = Recommender()
    silent = args.silent
    recommender.precision = args.precision
    recommender.run(args.r, args.t)
    

    
