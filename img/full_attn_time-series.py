import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# def noisy_periodic_func(x, amplitude=1, frequency=1, phase=0, noise_factor=1):
#     """Generate a noisy sinusoidal function"""
#     y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
#     noise = noise_factor * np.random.normal(size=len(x))
#     return y + noise

def noisy_cpu_usage_func(x, amplitude=1, frequency=1, phase=0, noise_factor=1):
    """Generate a noisy CPU usage-like function"""
    y = amplitude * np.abs(np.sin(2 * np.pi * frequency * x + phase))
    noise = noise_factor * np.random.normal(size=len(x))
    return y + noise


# Define parameters for the function and nodes
n_nodes = 6
n_points_func = 500
amplitude = 0.7
frequency = 1.3  # Lower frequency for more waves
noise_factor = 0.1  # Lower noise factor for less noise

# Generate x-coordinates for the function and the nodes
x_coords_func = np.linspace(0, n_nodes, n_points_func)
x_coords_nodes = np.arange(n_nodes)

# Generate y-coordinates for a less noisy function and the nodes
y_coords_func = noisy_cpu_usage_func(x_coords_func, amplitude=amplitude, frequency=frequency, noise_factor=noise_factor)
y_coords_nodes = noisy_cpu_usage_func(np.arange(n_nodes), amplitude=amplitude, frequency=frequency, noise_factor=noise_factor)

# Compute the minimum y-coordinate for the function (for drawing the dashed lines)
min_y_func = min(y_coords_func)

# Create a complete directed graph
G = nx.complete_graph(n_nodes, create_using=nx.DiGraph)

min_attention_strength = 0.3  # Minimum strength, between 0 and 1

# Assign weights to the edges
for (u, v, wt) in G.edges.data('weight'):
    if u == n_nodes - 1:  # Only the last node (T=now) has edges with weights other than 0
        wt = np.random.rand() * (1 - min_attention_strength) + min_attention_strength
    else:
        wt = 0
    G.edges[u, v]['weight'] = wt




# Extract the edges and weights from the graph
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

# Update positions for the nodes to lie on the function
pos = {i : (i, y_coords_nodes[i]) for i in range(n_nodes)}

# Set up plot
fig, ax = plt.subplots(figsize=(12, 6))

# Draw the function
plt.plot(x_coords_func, y_coords_func, color='lightgrey', linewidth=2, zorder=-10)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=1000)

# Draw edges with width corresponding to attention scores, only from the last node (T=now) to all others
# Make these edges dashed
width_factor = 5
edges_nonzero = [(u, v) for (u, v, wt) in G.edges.data('weight') if wt > 0]  # Only keep edges with non-zero weights
weights_nonzero = [wt for (u, v, wt) in G.edges.data('weight') if wt > 0]   # Only keep weights that are non-zero


# Drawing edges with corresponding colors and thickness
edge_colors = [G[u][v]['weight'] for u, v in edges_nonzero]  # Get weights for non-zero edges
edge_colors = plt.cm.Blues(np.array(edge_colors))  # Map weights to colors

# Normalize edge weights for thickness
edge_weights = np.array([G[u][v]['weight'] for u, v in edges_nonzero])
normalized_weights = 1 + 4 * (edge_weights - min(edge_weights)) / (max(edge_weights) - min(edge_weights))  # Scale between 1 and 5

nx.draw_networkx_edges(G, pos, edgelist=edges_nonzero, edge_color=edge_colors, style='dashed', width=normalized_weights)


#nx.draw_networkx_edges(G, pos, edgelist=edges_nonzero, width=np.array(weights_nonzero)*5, edge_color=weights_nonzero, edge_cmap=plt.cm.Blues, style='dashed', alpha=0.5, zorder=-1)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_color='black', font_size=24)

# Draw dots on the line where the dashed lines end without edgecolors
# for node, (x, y) in pos.items():
#     plt.scatter(x, min_y_func, c='white', s=300 * 1.5, zorder=3)  # Use filled circle marker without edgecolors, increase size by 1.5

# Draw circles at the end of each dashed line with the color corresponding to the strength of the connection
# for node, (x, y) in pos.items():
#     if G.has_edge(n_nodes - 1, node):  # Check if the edge from node 9 to this node exists
#         # Get the weight of the edge from the last node to this node
#         weight = G.get_edge_data(n_nodes - 1, node)['weight']

#         # Convert the weight to a color
#         color = plt.cm.Blues(weight)
#     else:
#         color = 'red'

#     plt.scatter(x, min_y_func, color=color, s=300 * 1.5, zorder=4, marker='o')  # Use filled circle marker, increase size by 1.5

# Draw the arrow indicating the flow of time
arrow_x = -0.5  # x position of the arrow, at the very left
arrow_y = min_y_func - 0.2  # y position of the arrow, a bit below the circles
arrow_dx = n_nodes + 0.5  # width of the arrow, as wide as the plot
arrow_dy = 0  # height of the arrow (0 because time flows horizontally)
ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.15, fc='black', ec='black')

# Add a "t" at the head of the arrow
text_x = n_nodes + 0.5  # x position of the text, a bit to the left of the arrow head
text_y = min_y_func - 0.22  # y position of the text, same as the arrow
ax.text(text_x, text_y, 't', fontsize=24)


# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Attention Strength', fontsize=15)

# Set title and remove axis
#plt.title("Self-Attention Mechanism in Transformer Model")
plt.axis('off')

# Save the figure as a PNG image
fig.savefig('./img/eps/full_attn_time-series.eps', format='eps', bbox_inches='tight')
plt.show()
