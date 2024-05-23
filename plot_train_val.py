import re
import matplotlib.pyplot as plt
import time
import os

# Initialize lists to store the losses
epochs = []
training_losses = []
validation_losses = []

# Define the log file path
# timestamp = time.strftime("%Y%m%d-%H%M%S")
# log_dir = "train_val_logs"
#log_file_path = os.path.join(log_dir, f"logs_{timestamp}.txt")
log_file_path = 'train_val_logs/logs_20240522-140542.txt'

# Regular expression pattern to match the log lines
pattern = r'Epoch \[(\d+)/\d+\], Training Loss: ([\d.]+), Validation Loss: ([\d.]+), Validation Accuracy: ([\d.]+)%'

# Read the log file and extract data
with open(log_file_path, 'r') as file:
    for line in file:
        match = re.match(pattern, line)
        if match:
            epoch = int(match.group(1))
            training_loss = float(match.group(2))
            validation_loss = float(match.group(3))
            epochs.append(epoch)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

# Print the extracted data (for verification)
print("Epochs:", epochs)
print("Training Losses:", training_losses)
print("Validation Losses:", validation_losses)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig('plots/train_val_plot.png')
#plt.show()
