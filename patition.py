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
  parser.add_argument('--data', type = str, default='ciao')
  parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
  args = parser.parse_args()
  num_partitions = 4
  device = torch.device(args.device)

  embed_dim = args.embed_dim

  path_data = 'data/' + args.data + ".pkl"
  data_file = open(path_data, 'rb')

  history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
    data_file)
