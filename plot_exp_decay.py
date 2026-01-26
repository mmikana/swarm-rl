import numpy as np
import matplotlib.pyplot as plt

# Define the exponential decay function
def exp_decay(k, x):
    return np.exp(-k * np.abs(x-1))

# Define x values from -1 to 1
x = np.linspace(-1, 1, 1000)

# Define multiple k values
k_values = [1.0, 2.0, 3.0, 4.0, 5.0]

# Create the plot
plt.figure(figsize=(12, 8))

for k in k_values:
    y = exp_decay(k, x)
    plt.plot(x, y, label=f'k={k}', linewidth=2)

plt.title(r'Exponential Decay Functions: $e^{-kx}$', fontsize=16)
plt.xlabel('x (meters)', fontsize=14)
plt.ylabel(r'$e^{-kx}$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([-1, 1])

# Add annotations for context
plt.text(-15, 0.8, 'Note: x represents distance in meters\nin the quadrotor environment', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/d/project/quad-swarm-rl/exp_decay_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'exp_decay_plot.png'")