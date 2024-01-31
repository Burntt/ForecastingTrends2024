import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create an 8x8 attention matrix with random values (including self-loops)
grid_size = 6
np.random.seed(42)
attention_matrix = np.random.rand(grid_size, grid_size)
np.fill_diagonal(attention_matrix, 1)

# Mask to keep only the lower triangular part and set the upper part to np.nan
lower_triangular_attention_matrix = np.tril(attention_matrix)
upper_mask = np.triu(np.ones_like(attention_matrix), k=1)
lower_triangular_attention_matrix[upper_mask == 1] = np.nan

# Normalize the lower triangular attention matrix
max_value = np.nanmax(lower_triangular_attention_matrix)
normalized_lower_triangular_attention_matrix = lower_triangular_attention_matrix / max_value

# Create an empty directed graph
G_explicit = nx.DiGraph()

# Add nodes
for i in range(grid_size):
    G_explicit.add_node(i)

# Add edges based on lower triangular matrix
for i in range(grid_size):
    for j in range(i + 1):  # Only loop up to i to keep lower triangular structure
        weight = normalized_lower_triangular_attention_matrix[i, j]
        if weight > 0:  # Only add an edge if the weight is greater than 0
            G_explicit.add_edge(i, j, weight=weight)

# Determine positions for the explicitly filtered graph
pos_explicit = nx.circular_layout(G_explicit)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the normalized lower triangular attention matrix
im = axes[0].imshow(normalized_lower_triangular_attention_matrix, cmap='Blues')
axes[0].set_title('Attention Matrix', fontsize=15)
axes[0].set_xlabel('Target Sequences', fontsize=15)
axes[0].set_ylabel('Source Sequences', fontsize=15)

# Set ticks for both axes to indicate 'blocks' from 0-4
axes[0].set_xticks(np.arange(-0.5, grid_size-0.5, 1), minor=True)
axes[0].set_yticks(np.arange(-0.5, grid_size-0.5, 1), minor=True)

# Set tick labels for both axes to indicate 'blocks' from 0-4
axes[0].set_xticks(np.arange(0, grid_size, 1))
axes[0].set_xticklabels(np.arange(0, grid_size, 1))
axes[0].set_yticks(np.arange(0, grid_size, 1))
axes[0].set_yticklabels(np.arange(0, grid_size, 1))

# Add grid
axes[0].grid(which='minor', color='black')


# Draw edges with matching colors, including self-loops
for i, j, weight in G_explicit.edges(data='weight'):
    color = plt.cm.Blues(weight)
    axes[1].plot(*zip(*[pos_explicit[i], pos_explicit[j]]), color=color, lw=7)

# Plot the graph, including self-loops
nx.draw_networkx_nodes(G_explicit, pos_explicit, node_size=700, node_color='skyblue', ax=axes[1])
nx.draw_networkx_labels(G_explicit, pos_explicit, font_size=15, font_color='black', ax=axes[1])

axes[1].set_title('Graph Visualization of Attention', fontsize=15)
axes[1].axis('off')

# Add colorbar to the right of the graph visualization with custom size and font size
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Attention Strength', fontsize=15)

plt.subplots_adjust(wspace=0.3)

fig.savefig('./img/pdf/full_attn_matrix_network.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

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
fig.savefig('./img/eps/full_attn_time-series.pdf', format='pdf', bbox_inches='tight')
plt.show()
