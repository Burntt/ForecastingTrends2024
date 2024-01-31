import matplotlib.pyplot as plt
import numpy as np

# Define the data
sequence_lengths = [12, 48, 96, 192, 288]
mse_scores = [2, 4, 5, 6.5, 8]
inference_speeds = [10**0, 10**-0.5, 10**-1, 10**-1.5, 10**-2]

# Define color and font preferences
color_blue = '#1f77b4'
color_red = '#d62728'
color_black = '#2c2c2c'

# Create the figure
fig, ax = plt.subplots(figsize=(8, 5.5))

# Plotting MSE scores
ax.plot(sequence_lengths, mse_scores, marker='o', label="MSE score", color=color_blue, linewidth=2.5, markersize=10)
ax.set_xlabel("Predict sequence length (5 min intervals)", fontsize=14)
ax.set_ylabel("MSE score", color=color_blue, fontsize=14)
ax.tick_params(axis='y', labelcolor=color_blue, labelsize=12, which='both', direction='in')
ax.tick_params(axis='x', labelsize=12, which='both', direction='in')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adding inference speeds on a second y-axis
ax2 = ax.twinx()
ax2.plot(sequence_lengths, inference_speeds, marker='s', label="Inference speed", linestyle="--", color=color_red, linewidth=2.5, markersize=10)
ax2.set_yscale('log')
ax2.set_ylabel("Predictions/sec (log scale)", color=color_red, fontsize=14)
ax2.tick_params(axis='y', labelcolor=color_red, labelsize=12, which='both', direction='in')
ax2.spines['top'].set_visible(False)

# Marking the point where performance deteriorates sharply
ax.plot(48, 4, marker='*', markersize=30, color=color_black, label="Deterioration point")

# Setting the legend on top of the figure
handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
fig.savefig('./img/eps/lstm_sequence_failure_mse.eps', format='eps', bbox_inches='tight')
plt.show()
