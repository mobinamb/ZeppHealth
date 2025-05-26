import matplotlib.pyplot as plt
import numpy as np
import os # Import os for saving the file

# Provided loss data
loss_data = """
Epoch 1/30, Average Loss: 0.912165
Epoch 2/30, Average Loss: 0.839281
Epoch 3/30, Average Loss: 0.824167
Epoch 4/30, Average Loss: 0.818249
Epoch 5/30, Average Loss: 0.812868
Epoch 6/30, Average Loss: 0.807449
Epoch 7/30, Average Loss: 0.806767
Epoch 8/30, Average Loss: 0.799156
Epoch 9/30, Average Loss: 0.794464
Epoch 10/30, Average Loss: 0.790098
Epoch 11/30, Average Loss: 0.786555
Epoch 12/30, Average Loss: 0.784584
Epoch 13/30, Average Loss: 0.779341
Epoch 14/30, Average Loss: 0.777672
Epoch 15/30, Average Loss: 0.775432
Epoch 16/30, Average Loss: 0.773390
Epoch 17/30, Average Loss: 0.772983
Epoch 18/30, Average Loss: 0.768346
Epoch 19/30, Average Loss: 0.766195
Epoch 20/30, Average Loss: 0.765035
Epoch 21/30, Average Loss: 0.766923
Epoch 22/30, Average Loss: 0.761370
Epoch 23/30, Average Loss: 0.760490
Epoch 24/30, Average Loss: 0.767114
Epoch 25/30, Average Loss: 0.766243
Epoch 26/30, Average Loss: 0.773546
Epoch 27/30, Average Loss: 0.763726
Epoch 28/30, Average Loss: 0.758100
Epoch 29/30, Average Loss: 0.756026
Epoch 30/30, Average Loss: 0.754393
"""

# Parse the data
epochs = []
losses = []
for line in loss_data.strip().split('\n'):
    parts = line.split(',')
    
    # Get the part like "1/30"
    epoch_fraction_str = parts[0].split(' ')[1] 
    
    # Split "1/30" by '/' to get just the "1"
    current_epoch = int(epoch_fraction_str.split('/')[0]) # Corrected line
    
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