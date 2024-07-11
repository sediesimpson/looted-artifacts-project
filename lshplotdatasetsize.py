import numpy as np
import matplotlib.pyplot as plt
import os
import sys

    
# Define the directory where the .npy files are saved
save_dir = 'datasetsizes/'

# Load the dataset sizes and query times from the .npy files
dataset_sizes = np.load(os.path.join(save_dir, "dataset_sizes_memory.npy"))
#query_times = np.load(os.path.join(save_dir, "lsh_subset_14039.npy"))
query_times = np.load('lsh_subset_1000.npy', allow_pickle=True)
query_times = query_times / (1024 * 1024)
# Print raw data
print("Dataset Sizes:", dataset_sizes)
print("Memory Usages (MB):", query_times)
sys.exit()

# print(dataset_sizes)
# print(query_times)
# sys.exit()
# Plot Query Time vs Dataset Size
plt.figure(figsize=(10, 5))
plt.plot(dataset_sizes, query_times, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Memory Usage (Megabytes)')
plt.title('Memory Usage vs Dataset Size')
plt.grid(True)
#plt.show()
plt.savefig('datasetsizes/memory.png')
