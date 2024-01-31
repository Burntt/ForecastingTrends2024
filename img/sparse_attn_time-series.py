import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# Define the active nodes and their colors
active_nodes = [1, 3, 5]  # Chosen randomly, you can define your own active nodes
active_color = 'red'
inactive_color = 'black'


min_attention_strength = 0.6  # Minimum strength, between 0 and 1

# Assign weights to the edges
for (u, v, wt) in G.edges.data('weight'):
    if u == n_nodes - 1:  # Only the last node (T=now) has edges with weights other than 0
        wt = np.random.rand() * (1 - min_attention_strength) + min_attention_strength
    else:
        wt = 0
    G.edges[u, v]['weight'] = wt


# Update positions for the nodes to lie on the function
pos = {i : (i, y_coords_nodes[i]) for i in range(n_nodes)}

# Set up plot
fig, ax = plt.subplots(figsize=(12, 6))

# Draw the function
plt.plot(x_coords_func, y_coords_func, color='lightgrey', linewidth=2, zorder=-10)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=1000)

# Draw edges with width and color corresponding to attention scores, only from the last node (T=now) to all others
# Make these edges dashed
width_factor = 5
edges_nonzero = [(u, v) for (u, v, wt) in G.edges.data('weight') if wt > 0]  # Only keep edges with non-zero weights
edge_colors = [active_color if v in active_nodes else inactive_color for u, v in edges_nonzero]
edge_widths = [G[u][v]['weight'] * width_factor for u, v in edges_nonzero]
nx.draw_networkx_edges(G, pos, edgelist=edges_nonzero, width=edge_widths, edge_color=edge_colors, style='dashed')

# Find the edges that are active (red) and have non-zero weights
active_edges = [(u, v) for (u, v, wt) in G.edges.data('weight') if wt > 0 and v in active_nodes]

# Update edge widths based on these active edges
active_edge_widths = [G[u][v]['weight'] * width_factor for u, v in active_edges]

# Replot the graph, but this time only include the active edges
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(x_coords_func, y_coords_func, color='lightgrey', linewidth=2, zorder=-10)
nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=1000)
nx.draw_networkx_edges(G, pos, edgelist=active_edges, width=active_edge_widths, edge_color=active_color, style='dashed')
nx.draw_networkx_labels(G, pos, font_color='black', font_size=24)

# Additional plot features
for node, (x, y) in pos.items():
    plt.plot([x, x], [y, min_y_func], color='grey', zorder=1, linestyle=(0, (5, 1)))

arrow_x = -0.5
arrow_y = min_y_func - 0.2
arrow_dx = n_nodes + 0.5
arrow_dy = 0
ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.15, fc='black', ec='black')

sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Attention Strength', fontsize=15)

plt.axis('off')
fig.savefig('./img/eps/sparse_attn_time-series.eps', format='eps', bbox_inches='tight')
plt.show()
