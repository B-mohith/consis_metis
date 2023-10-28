import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_graphs, load_graphs
from dgl import DGLGraph
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.metis import partition

from Node_Encoders import Node_Encoder
from Node_Aggregators import Node_Aggregator
from GraphConsis import GraphConsis

# Define a custom dataset class for your data (you may need to adjust this)
class CustomDataset(DGLDataset):
    def __init__(self, name=None):
        super(CustomDataset, self).__init__(name=name)

    def process(self):
        # Load and preprocess your data
        # Create DGL graphs for users and items
        self.graphs = [user_graph, item_graph]  # Replace with your own data

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

# Add the graph partitioning function
def metis_partition(graph, num_partitions):
    # Convert DGL graph to NetworkX for partitioning
    import networkx as nx
    g_nx = graph.to_networkx(node_attrs=None, edge_attrs=None)

    # Perform Metis graph partitioning
    _, parts = partition.part_graph(g_nx, num_partitions, recursive=True)
    
    # Create a mapping from node IDs to partition IDs
    node_to_partition = {}
    for node_id, partition_id in enumerate(parts):
        node_to_partition[node_id] = partition_id

    return node_to_partition

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphConsis model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--percent', type=float, default=0.4, help='neighbor percent')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='Load from checkpoint or not')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--data', type = str, default='ciao')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--num_partitions', type=int, default=4, help='Number of partitions')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(args.device)

    embed_dim = args.embed_dim

    path_data = 'data/' + args.data + ".pkl"
    data_file = open(path_data, 'rb')

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(data_file)

    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)

    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]

    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]

    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]

    # Create DGL graphs for users and items (replace with your own data)
    user_graph = DGLGraph()
    user_graph.add_nodes(num_users)
    user_graph.add_edges(train_u, train_v)
    user_graph.ndata['user_id'] = torch.LongTensor(train_u)
    user_graph.ndata['ratings'] = torch.FloatTensor(train_r)

    item_graph = DGLGraph()
    item_graph.add_nodes(num_items)
    item_graph.add_edges(train_v, train_u)
    item_graph.ndata['item_id'] = torch.LongTensor(train_v)

    # Partition the user and item graphs
    user_partitions = metis_partition(user_graph, args.num_partitions)
    item_partitions = metis_partition(item_graph, args.num_partitions)

    # Save the partitioned graphs using DGL's save_graphs
    save_graphs('partitioned_data/user_graph', [user_graph], user_partitions)
    save_graphs('partitioned_data/item_graph', [item_graph], item_partitions)

    # Load the partitioned graphs
    user_graphs, _ = load_graphs('partitioned_data/user_graph/0.bin')
    item_graphs, _ = load_graphs('partitioned_data/item_graph/0.bin')

    # Define data loaders
    train_loader = GraphDataLoader(user_graphs, torch.arange(len(train_u)), batch_size=args.batch_size, shuffle=True)

    # Define your user, item, and rating embeddings
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)

    # Define the rest of your model (Node_Aggregator, Node_Encoder, and GraphConsis)

    # ...

    # Perform training and testing using partitioned graphs
    # Adjust your code accordingly to work with partitioned graphs

if __name__ == "__main__":
    main()
