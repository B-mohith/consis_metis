import pickle
import matplotlib.pyplot as plt
import networkx as nx

# Load the pickle file
path_data = 'data/' + 'filmtrust' + ".pkl"
data_file = open(path_data, 'rb')
history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
    data_file)

# Create a graph object
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

# Plot the graph
plt.figure(figsize=(10, 8))
nx.draw(graph, with_labels=True)
plt.title("filmtrust Dataset Graph")
plt.show()
