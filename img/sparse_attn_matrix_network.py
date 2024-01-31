import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create random sparse attention matrix
def create_randomized_sparse_attention_matrix(size, variance=0.2):
    attention_matrix = np.zeros((size, size))
    
    for i in range(size):
        attention_matrix[i, i] = 1
        if i > 0:
            attention_matrix[i, i - 1] = np.random.normal(0.7, variance)
        if i < size - 1:
            attention_matrix[i, i + 1] = np.random.normal(0.7, variance)
        if i > 1:
            attention_matrix[i, i - 2] = np.random.normal(0.4, variance)
        if i < size - 2:
            attention_matrix[i, i + 2] = np.random.normal(0.4, variance)

    return attention_matrix

grid_size = 6
sparse_attention_matrix = create_randomized_sparse_attention_matrix(grid_size)

# Mask to make non-relevant parts NaN and enforce lower-triangular structure
for i in range(grid_size):
    for j in range(grid_size):
        if abs(i - j) > 2 or i < j:
            sparse_attention_matrix[i, j] = np.nan


# Normalize the matrix
max_value = np.nanmax(sparse_attention_matrix)
normalized_sparse_attention_matrix = sparse_attention_matrix / max_value

# Create graph
G_sparse_explicit = nx.DiGraph()

# Add nodes
for i in range(grid_size):
    G_sparse_explicit.add_node(i)

# Add edges
for i in range(grid_size):
    for j in range(i + 1):  # Only consider lower triangular part
        weight = sparse_attention_matrix[i, j] / max_value  # Normalize
        if not np.isnan(weight):
            G_sparse_explicit.add_edge(i, j, weight=weight)


# Layout
pos_sparse_explicit = nx.circular_layout(G_sparse_explicit)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the normalized lower triangular attention matrix
im = axes[0].imshow(normalized_sparse_attention_matrix, cmap='Reds')
axes[0].set_title('Sparse Attention Matrix', fontsize=15)
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

# Graph
for i, j, weight in G_sparse_explicit.edges(data='weight'):
    axes[1].plot(*zip(*[pos_sparse_explicit[i], pos_sparse_explicit[j]]), color=plt.cm.Reds(weight), lw=7)

nx.draw_networkx_nodes(G_sparse_explicit, pos_sparse_explicit, node_size=700, node_color='lightcoral', ax=axes[1])
nx.draw_networkx_labels(G_sparse_explicit, pos_sparse_explicit, font_size=15, ax=axes[1])

axes[1].set_title('Graph Visualization of Sparse Attention', fontsize=15)
axes[1].axis('off')

# Colorbar
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Attention Strength')
fig.savefig('./img/pdf/sparse_attn_matrix_network.pdf', format='pdf', bbox_inches='tight')


plt.show()
