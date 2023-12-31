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
import metis
import networkx as nx
import metis

def partition_graph(graph, num_partitions):  
  metis_graph = metis.networkx_to_metis(graph)

  # Partition the graph using Metis.
  edgecuts, parts = metis.part_graph(metis_graph, nparts=num_partitions)

  # Split the graph into subgraphs based on the partitions.
  subgraphs = [nx.Graph() for _ in range(num_partitions)]
  for node, part in enumerate(parts):
    subgraphs[part].add_node(node)

  for edge in graph.edges():
    node_u, node_v = edge
    part_u = parts[node_u]
    part_v = parts[node_v]

    if part_u == part_v:
      subgraphs[part_u].add_edge(node_u, node_v)

  return subgraphs


import networkx as nx

def construct_social_recommendation_graph(history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists): 

  graph = nx.Graph()

  # Add the users and the items to the graph.
  for user_id in range(len(history_u_lists)):
    graph.add_node(user_id, type="user")
  for item_id in range(len(history_v_lists)):
    graph.add_node(item_id, type="item")

  # Add the edges to the graph.
  for user_id in range(len(history_u_lists)):
    for item_id in history_u_lists[user_id]:
      graph.add_edge(user_id, item_id, type="interaction")
    for other_user_id in social_adj_lists[user_id]:
      graph.add_edge(user_id, other_user_id, type="social")

  for item_id in range(len(history_v_lists)):
    for user_id in history_v_lists[item_id]:
      graph.add_edge(user_id, item_id, type="interaction")
    for other_item_id in item_adj_lists[item_id]:
      graph.add_edge(item_id, other_item_id, type="item")

  return graph


import torch

def create_data_loader(subgraph, batch_size):
    
    edges = list(subgraph.edges())
    num_edges = len(edges)
    num_batches = num_edges // batch_size + 1

    # Create data for the data loader
    data = [(edges[i:i+batch_size]) for i in range(0, num_edges, batch_size)]

    # Define a custom DataLoader for the subgraph
    class SubgraphDataLoader(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    subgraph_loader = torch.utils.data.DataLoader(SubgraphDataLoader(data), batch_size=1, shuffle=True)

    return subgraph_loader

def train(model, device, social_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    labels_list = torch.tensor([]).to(device)
  
    for i, data in enumerate(social_loader, 0):
      #print(data)
        # Unpack the data in the correct format
      batch_nodes_u, *batch_nodes_v = data
      #labels_list = None
      batch_nodes_u = torch.LongTensor(batch_nodes_u).to(device)
      batch_nodes_v = torch.LongTensor(batch_nodes_v).to(device)

      optimizer.zero_grad()
      loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 100 == 0:
          print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
               epoch, i, running_loss / 100, best_rmse, best_mae))
          running_loss = 0.0
    return 0



'''    
def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0
    '''


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

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
    args = parser.parse_args()
    num_partitions = 4

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(args.device)

    embed_dim = args.embed_dim

    path_data = 'data/' + args.data + ".pkl"
    data_file = open(path_data, 'rb')

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
        data_file)
    
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

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)   
    
    graph_input = construct_social_recommendation_graph(history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists)
    social_subgraphs = partition_graph(graph_input, 4)
   

    for i in range(num_partitions):
        social_subgraph = social_subgraphs[i]
        #item_subgraph = item_subgraphs[i]

        # Create data loaders for the current subgraph
        social_loader = create_data_loader(social_subgraph, args.batch_size)
        #item_loader = create_data_loader(item_subgraph, args.batch_size)
        
        
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)
    #node_feature
    node_agg = Node_Aggregator(v2e, r2e, u2e, embed_dim, r2e.num_embeddings - 1, cuda=device)
    node_enc = Node_Encoder(u2e, v2e, embed_dim, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists, node_agg, percent=args.percent,  cuda=device)

    # model
    graphconsis = GraphConsis(node_enc, r2e).to(device)
    optimizer = torch.optim.Adam(graphconsis.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    # load from checkpoint
    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.data + '.pt')
        graphconsis.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphconsis, device, social_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphconsis, device, valid_loader)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': graphconsis.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.data + '.pt')
        else:
            endure_count += 1
        print("rmse on valid set: %.4f, mae:%.4f " % (expected_rmse, mae))
        rmse, mae = test(graphconsis, device, test_loader)
        print('rmse on test set: %.4f, mae:%.4f '%(rmse, mae))

        if endure_count > 5:
            break
    print('finished')


if __name__ == "__main__":
    main()
