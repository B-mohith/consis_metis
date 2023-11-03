import pickle
import matplotlib.pyplot as plt

# Load the pickle file
with open("data/filmtrust.pkl", "rb") as f:
    dataset = pickle.load(f)

# Create a graph object
G = nx.Graph()

# Add the nodes and edges to the graph
for node in dataset["nodes"]:
    G.add_node(node)

for edge in dataset["edges"]:
    G.add_edge(edge[0], edge[1])

# Plot the graph
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True)
plt.title("filmtrust Dataset Graph")
plt.show()
