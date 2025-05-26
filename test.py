import matplotlib.pyplot as plt
import numpy as np
import os # Import os for saving the file

# Provided loss data
loss_data = """
Epoch 1/5, Average Loss: 0.729995
Epoch 2/5, Average Loss: 0.647050
Epoch 3/5, Average Loss: 0.588682
Epoch 4/5, Average Loss: 0.546123
Epoch 5/5, Average Loss: 0.515123
"""

# Parse the data
epochs = []
losses = []
for line in loss_data.strip().split('\n'):
    parts = line.split(',')
    
    # Get the part like "1/30"
    epoch_fraction_str = parts[0].split(' ')[1] 
    
    # Split "1/30" by '/' to get just the "1"
    current_epoch = int(epoch_fraction_str.split('/')[0]) 
    
    loss_str = parts[1].split(': ')[1] 
    
    epochs.append(current_epoch)
    losses.append(float(loss_str))

# Create the plot
plt.figure(figsize=(10, 6)) # Set figure size for better readability
plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Training Loss')

# Add titles and labels
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True) # Add a grid for easier reading

# Set x-axis ticks at intervals of 5 epochs, ensuring they cover the full range
plt.xticks(np.arange(min(epochs), max(epochs)+1, 5)) 

# Adjust y-ticks for better granularity, extending slightly beyond min/max losses
# np.ceil and np.floor are used to ensure ticks cover the full range
y_min = np.floor(min(losses) * 100) / 100 - 0.01 # Round down to nearest 0.01 and subtract a bit
y_max = np.ceil(max(losses) * 100) / 100 + 0.01  # Round up to nearest 0.01 and add a bit
plt.yticks(np.arange(y_min, y_max, 0.01)) 

plt.legend() # Show the legend

# Save the plot to a file
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
plot_filename = os.path.join(output_dir, 'loss_vs_epoch_plot.png')
plt.savefig(plot_filename)

print(f"Plot saved successfully to {plot_filename}")

# Display the plot (optional, will only show if running in a graphical environment)
plt.show()