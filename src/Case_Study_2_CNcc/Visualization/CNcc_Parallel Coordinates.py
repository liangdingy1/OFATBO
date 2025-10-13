import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# Read data
df = pd.read_csv('bb_30_CNcc_beta_acq_Radial_Visualization.csv')
params = ['t', 'p', 'cat', 'arbr', 'pyrr', 'yields']

# Define parameter ranges based on your reference
param_ranges = {
    't': (5, 12),
    'p': (19.5, 24.0),
    'cat': (0.00001, 0.00010),
    'arbr': (0.15, 0.5),
    'pyrr': (1.0, 2.0),
    'yields': (0, 100)
}

# Create figure
plt.figure(figsize=(16, 10))
ax = plt.gca()

# Create a list to store the scaled values
scaled_values = []

# Scale each parameter to [0,1] range based on its own range
for param in params:
    min_val, max_val = param_ranges[param]
    scaled = (df[param] - min_val) / (max_val - min_val)
    scaled_values.append(scaled.values)

# Convert to numpy array
scaled_values = np.array(scaled_values).T

# Plot all experiments with gray lines
for i in range(len(df)):
    plt.plot(range(len(params)), scaled_values[i],
             color='gray', linewidth=0.5, alpha=0.3)

# Highlight best experiment
best_idx = df['yields'].argmax()
plt.plot(range(len(params)), scaled_values[best_idx],
         'o-', color='#FF6B6B', linewidth=3, markersize=8,
         label=f'Best Yield: {df["yields"].iloc[best_idx]:.2f}%')

# Highlight worst experiment
worst_idx = df['yields'].argmin()
plt.plot(range(len(params)), scaled_values[worst_idx],
         's--', color='#4ECDC4', linewidth=2, markersize=8,
         label=f'Worst Yield: {df["yields"].iloc[worst_idx]:.2f}%')

# Customize axes
plt.xticks(range(len(params)), params, fontsize=12, weight='bold')

# Create custom y-axis labels showing original values
for i, param in enumerate(params):
    min_val, max_val = param_ranges[param]

    # Create 5 ticks between min and max
    ticks = np.linspace(0, 1, 5)
    tick_labels = []

    for tick in ticks:
        # Convert back to original scale
        value = min_val + tick * (max_val - min_val)

        # Format based on parameter magnitude
        if param == 'cat':
            tick_labels.append(f'{value:.5f}')
        elif param in ['arbr', 'pyrr']:
            tick_labels.append(f'{value:.2f}')
        elif param == 'yields':
            tick_labels.append(f'{value:.1f}%')
        else:
            tick_labels.append(f'{value:.1f}')

    # Set the tick positions and labels for each parameter
    ax.get_yaxis().set_ticks(ticks)
    ax.get_yaxis().set_ticklabels(tick_labels)

    # Only show the grid for the current parameter
    ax.yaxis.grid(True, alpha=0.3)

    # # Move to next parameter
    # if i < len(params) - 1:
    #     ax = ax.twinx()
    #     ax.set_ylim(0, 1)
    #     ax.spines['right'].set_position(('axes', (i + 1) / len(params)))

# Set the first y-axis to be visible
plt.gca().yaxis.set_visible(True)

# Add legend and title
plt.legend(loc='upper right', fontsize=11)
plt.title('Parallel Coordinates Plot - Independent Parameter Scaling',
          fontsize=16, weight='bold', pad=20)
plt.xlabel('Parameters', fontsize=13, weight='bold', labelpad=15)

# Adjust layout
plt.tight_layout()
plt.savefig('parallel_coordinates_independent_scaling.png', dpi=300, bbox_inches='tight')
plt.show()

# Print parameter ranges for reference
print("Parameter Ranges Used:")
for param in params:
    min_val, max_val = param_ranges[param]
    print(f"{param}: {min_val} - {max_val}")