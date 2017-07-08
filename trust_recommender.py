import networkx as nx
import igraph
from girvan_newman import girvan_newman
from fix_dendrogram import fix_dendrogram
from collections import defaultdict

# Compiled Infomap.
from infomap import infomap

from recommender import Recommender
from utils import log

    
class TrustRecommender(Recommender):
    def __init__(self):
        super(TrustRecommender, self).__init__()

        self.trust_net = None
        self.shortest_paths = {}
        self.trust_net_path = None

    def prepare(self):
        assert self.trust_net_path

        #self.trust_net = nx.read_edgelist(
        #    self.trust_net_path,
        #    create_using=nx.DiGraph(),
        #    nodetype=int
        #)

        self.trust_net = igraph.Graph.Read_Edgelist(self.trust_net_path, directed=True)
        
        super(TrustRecommender, self).prepare()
                
    def _cluster_users(self):
        assert self.trust_net

        dendrogram = self.trust_net.community_edge_betweenness(10) #community_leading_eigenvector()
        fix_dendrogram(self.trust_net, dendrogram)

        self.user2cluster = defaultdict(str)
        self.cluster2users = defaultdict(set)

        for i in range(self.trust_net.vcount()):
            cl = dendrogram.as_clustering(i + 1)
            for i, c in enumerate(cl):
                com_id = '{}:{}'.format(self.user2cluster[c[0]], i)
                com_id = '0' if com_id == ':0' else com_id
                
                for u in c:
                    self.user2cluster[u] = com_id
                    self.cluster2users[com_id].add(u)


        with open('./trust_data/user2clusters.dat', 'w') as out:
            for u in self.user2cluster:
                print(len(self.user2cluster[u]))
                out.write('{}-'.format(u))
                out.write(self.user2cluster[u])
                out.write('\n')
                
        with open('./trust_data/cluster2users.dat', 'w') as out:
            for c in self.cluster2users:
                out.write('{}-'.format(c))
                for u in self.cluster2users[c]:
                    out.write('{}:'.format(u))
                out.write('\n')
                
        self.trust_net = nx.read_edgelist(
            self.trust_net_path,
            create_using=nx.DiGraph(),
            nodetype=int
        )
        
        return self.user2cluster
                             
    
    def _com_queue(self, user):
        assert self.user2cluster
        assert user in self.user2cluster
        
        com_id = self.user2cluster[user]
        com_queue = []
        
        while True:
            com_queue.append(com_id)
            com_id = com_id.rsplit(':', 1)[0]
            
            if not ':' in com_id:
                com_queue.append(com_id)
                break

        return com_queue

    def _com2nodes(self, com):
        pass
    
    def predict(self, user, item):
        assert self.ratings
        assert self.trust_net
        assert self.avg_ratings
        assert self.min_nraters
        assert type(self.shortest_paths) == dict
        
        delta = 0
        weights = []
        path_lengths = []
        paths_from_user = None
        weight_sum = 0.00000001
        com_queue = self._com_queue(user)
        avg_rating = self.avg_ratings.get(user, self.global_avg_rating)

        for com_id in com_queue:
            ratings = []

            #print(com_queue, com_id, len(self.cluster2users[com_id]))
            
            for rater in (r for r in self.cluster2users[com_id] if r != user):
                if rater in self.ratings and item in self.ratings[rater]:
                    ratings.append((rater, self.ratings[rater][item]))

            if ratings:
                if not com_id in self.shortest_paths:
                    self.shortest_paths[com_id] = {}

                if not user in self.shortest_paths[com_id]:
                    subnet = self.trust_net.subgraph(self.cluster2users[com_id])
                    self.shortest_paths[com_id][user] = nx.single_source_shortest_path(subnet, user) 

                paths_from_user = self.shortest_paths[com_id][user]
                break

        if not ratings:
            return -1 #self._round(avg_rating)

        if len(ratings) == 1:
            delta = ratings[0][1] - self.avg_ratings[ratings[0][0]]
            return self._round(avg_rating + delta)
        
        assert paths_from_user
            
        for rater, rating in ratings:
            if rater in paths_from_user:
                path_lengths.append(len(paths_from_user[rater]) - 1)
            else:
                path_lengths.append(1000000)

        total_path_length = sum(path_lengths)

        for i in range(len(ratings)):
            weights.append(1 - (float(path_lengths[i]) / total_path_length))                    
            weight_sum += weights[i]
            
        i = 0
        for rater, rating in ratings:
            change = float(weights[i] * (rating - self.avg_ratings[rater])) / weight_sum
            delta += change
            i += 1

        return min(5, max(0, self._round(avg_rating + delta)))
    

if __name__ == '__main__':
    recommender = TrustRecommender()
    parser = recommender._get_parser()

    parser.add_argument(
        'trust_net_path',
        type=str,
        help='Path to trust network.'
    )
    
    args = parser.parse_args()
    
    recommender.init(args)

    recommender.trust_net_path = args.trust_net_path
    
    recommender.prepare()
    recommender.test()
