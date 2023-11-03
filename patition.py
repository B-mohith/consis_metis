import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from Node_Encoders import Node_Encoder
from Node_Aggregators import Node_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import sys
from GraphConsis import GraphConsis
import networkx as nx
import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser(description='Social Recommendation: GraphConsis model')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
  parser.add_argument('--percent', type=float, default=0.4, help='neighbor percent')
  parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
  parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
  parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
  parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
  parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='Load from checkpoint or not')
  parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
  parser.add_argument('--data', type = str, default='filmtrust')
  parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
  args = parser.parse_args()
  num_partitions = 4
  device = torch.device(args.device)

  embed_dim = args.embed_dim

  path_data = 'data/' + args.data + ".pkl"
  data_file = open(path_data, 'rb')

  history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
    data_file)
  G = nx.Graph()

# Add the users and items as nodes to the graph.
  for u in history_u_lists:
    print("1")
    G.add_node(u, type='user')

  for v in history_v_lists:
    print("2")
    G.add_node(v, type='item')

# Add the edges to the graph.
  for u, ur_list in history_ur_lists.items():
    for i in ur_list:
        print("3")
        G.add_edge(u, i, weight=1)

  for v, vr_list in history_vr_lists.items():
    for r in vr_list:

        G.add_edge(r, v, weight=1)

  pos = nx.fruchterman_reingold_layout(G)

# Draw the nodes and edges of the graph.
  nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.5)
  nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.3)

# Add labels to the nodes.
  for node, attr in G.nodes(data=True):
    if attr['type'] == 'user':
        label = 'User ' + str(node)
    else:
        label = 'Item ' + str(node)
    
    nx.draw_networkx_labels(G, pos, labels={node: label}, horizontalalignment='center', verticalalignment='center')

# Set the plot title and axis labels.
  plt.title('Ciao Dataset Graph')
  plt.xlabel('Users')
  plt.ylabel('Items')

# Show the plot.
  print("6")
  plt.show()
if __name__ == "__main__":
  main()
