import os
import sys
from tqdm import tqdm
import numpy as np

from knndataloader2 import *


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
from torchvision import models, transforms


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from sklearn.metrics import precision_score, recall_score, f1_score, pairwise_distances
from sklearn.utils import shuffle

import math
import seaborn as sns

from pprint import pprint
import torch.optim as optim
#------------------------------------------------------------------------------------------------------------------------
# Define CustomResNetClassifier
#------------------------------------------------------------------------------------------------------------------------
class CustomResNetClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, weights=None):
        super(CustomResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights=weights)
        self.resnet.requires_grad_(False)  # Freeze all layers

        # Replace the fully connected layer with a custom sequential layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes)  # Output layer with one neuron per class for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

#------------------------------------------------------------------------------------------------------------------------
# Define Model
#------------------------------------------------------------------------------------------------------------------------
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = CustomResNetClassifier(weights=weights, num_classes=28, hidden_dim=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
batch_size = 32
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Create train dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/duplicatedata"
train_dataset = CustomImageDatasetTrain(root_dir)
test_dataset = CustomImageDatasetTest(root_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#--------------------------------------------------------------------------------------------------------------------------
# Load checkpoints from trained model with updated weights
#--------------------------------------------------------------------------------------------------------------------------
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print('Epoch of model loaded:', epoch)
print('Loss of model loaded:', loss)
#------------------------------------------------------------------------------------------------------------------------
# Define the feature extraction function
#------------------------------------------------------------------------------------------------------------------------
def extract_features(dataloader, model, device):
    model.eval()
    features_list = []
    labels_list = []
    label_indices_list = []
    img_paths_list = []
    label_counts_list = []
    
    with torch.no_grad():
        for imgs, labels, labels_idx, img_paths, label_counts in tqdm(dataloader, desc="Extracting features"):
            # Move images to the specified device
            imgs = imgs.to(device)
            
            # Extract features
            features = model(imgs)
            #print('feature dimensions:',features.shape)
            
            # Append features and labels to the respective lists
            features_list.append(features.cpu().numpy())
            label_indices_list.append(labels_idx.cpu().numpy())
            labels_list.extend(labels)
            img_paths_list.extend(img_paths)
            label_counts_list.extend(label_counts)
 

    # Flatten the list of arrays (handle varying batch sizes)
    features_array = np.concatenate(features_list, axis=0)
    labels_indices_array = np.concatenate(label_indices_list, axis=0)

    
    return features_array, labels_indices_array, img_paths_list, labels_list, label_counts_list

#------------------------------------------------------------------------------------------------------------------------
# Generate features
#------------------------------------------------------------------------------------------------------------------------
# Extract features for training and validation sets
train_features, train_label_indices, train_img_paths, train_labels, train_label_counts = extract_features(train_loader, model, device)
test_features, test_label_indices, test_img_paths, test_labels, test_label_counts = extract_features(test_loader, model, device)

#------------------------------------------------------------------------------------------------------------------------
# Test image examples
#------------------------------------------------------------------------------------------------------------------------
query_index = 80
nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.008).fit(test_features)
distances, indices = nbrs.radius_neighbors([test_features[query_index]], sort_results=True)
print(len(distances[0]))

print('===========  INDICES ============')
print(indices)
print()
print('=========== DISTANCES ============')
print(distances)
print()
print('=========== TEST IMAGE PATH ============')
imgs_to_plot = []
print(test_img_paths[query_index])
print()
print('=========== NEIGHBOUR IMAGE PATHS ============')
for i in indices[0]:
    imgs_to_plot.append(test_img_paths[i])
print(imgs_to_plot)

# print('=========== PLOTTING IMAGES ============')
# # Number of images
# num_images = len(imgs_to_plot)

# # Maximum number of columns
# max_cols = 5
# rows = math.ceil(num_images / max_cols)  # Calculate the number of rows needed

# # Create a figure with a dynamic grid of subplots (multiple rows if needed)
# fig, axes = plt.subplots(rows, min(num_images, max_cols), figsize=(min(num_images, max_cols) * 5, rows * 5))  # Adjust figsize as needed

# # Flatten axes array for easier iteration if grid has more than 1 row
# axes = axes.flatten() if num_images > 1 else [axes]

# # Loop through each image path and plot them
# for i, img_path in enumerate(imgs_to_plot):
#     img = Image.open(img_path)  # Open the image
#     axes[i].imshow(img)  # Display the image
#     axes[i].axis('off')  # Hide the axes

#     # Add labels
#     if i == 0:
#         axes[i].set_title("Query Image")
#     else:
#         axes[i].set_title(f"Neighbour {i}")

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')

# # Show the plot with all images
# plt.savefig('Aplotsknn/new_knn_22.pdf')

# print('=========== PRECISION AND RECALL FOR QUERY IMAGE ============')

# true_label = test_labels[query_index]
# true_label_idx = test_label_indices[query_index]
# true_label_counts = test_label_counts[query_index].item()
# neighbor_labels = test_label_indices[indices[0]]


# tp = 0
# fp = 0
# tn = 0
# fn = 0
# for label in neighbor_labels:
#     print(f"neighbour label: {label}, true label: {true_label_idx}")
#     if label == true_label_idx:
#         tp += 1
#     else:
#         fp += 1

# fn = true_label_counts - tp
# tn = len(test_features) - len(neighbor_labels) - fn

# precision = tp / (tp + fp)
# recall = tp / (tp + fn)

# print(f"precision={precision}, recall={recall}") 

# #------------------------------------------------------------------------------------------------------------------------
# Fit the nearest neighbours to test set
# #------------------------------------------------------------------------------------------------------------------------
# true_positives = 0
# false_positives = 0
# true_negatives = 0
# false_negatives = 0 
# precision = 0
# recall = 0

# nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.35).fit(test_features)

# for query_index in range(len(test_features)):
#     distances, indices = nbrs.radius_neighbors([test_features[query_index]], sort_results=True)

#     distances = distances[0] # remove first distance as it is itself
#     indices = indices[0] # remove first index as it is itself

#     true_label = test_labels[query_index]
#     true_label_idx = test_label_indices[query_index]
#     true_label_counts = test_label_counts[query_index].item()

#     neighbor_labels = test_label_indices[indices]

#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#     for label in neighbor_labels:
#         print(f"neighbour label: {label}, true label: {true_label_idx}")
#         if label == true_label_idx:
#             tp += 1
#         else:
#             fp += 1

#     fn = true_label_counts - tp
#     tn = len(test_features) - len(neighbor_labels) - fn

#     # accumulate
#     true_positives += tp
#     false_positives += fp
#     false_negatives += fn
#     true_negatives += tn

# if (true_positives + false_negatives) > 0:
#     recall = true_positives / (true_positives + false_negatives)
# else:
#     recall = 0

# if (true_positives + false_positives) > 0:
#     precision = true_positives / (true_positives + false_positives) 
# else:
#     precision = 0

# print("Overall results")
# results = {
#     "true positives": true_positives,
#     "false positives": false_positives,
#     "true negatives": true_negatives,
#     "false negatives": false_negatives,
#     "recall": recall,
#     "precision": precision
# }
# pprint(results)


# #------------------------------------------------------------------------------------------------------------------------
# # Find the optimal radius
# #------------------------------------------------------------------------------------------------------------------------
# train_labels = np.array(train_labels)
# # Since image embeddings are in a matrix (n_samples, n_features)
# # 1. Compute cosine distances between all images in the dataset
# distances = cosine_distances(train_features) # N x N distance matrix

# # 2. For each image, exclude self-distances (set diagonal to a large value)
# np.fill_diagonal(distances, np.inf)  # Set self-distances to infinity

    
# # 3. Function to get top-k neighbors for 90% recall for any image
# def get_max_distance(query_index, recall_target=0.5):
#     true_label = train_labels[query_index]

#     # Sort distances and corresponding labels
#     sorted_index = np.argsort(distances[query_index])
#     sorted_index = np.array(sorted_index)
#     sorted_distances = distances[query_index][sorted_index]  # Sorted distances

#     # Exclude `inf` values from the sorted distances
#     valid_distances = sorted_distances[sorted_distances != np.inf]

#     # Identify the relevant neighbors (images with the same label as the query)
#     sorted_labels = train_labels[sorted_index]

#     # Get all neighbors that match the query's label
#     relevant = np.where(sorted_labels == true_label)[0]
    
#     # Determine number of true positives needed for 90% recall
#     # `k_needed` is the minimum number of relevant neighbors you must retrieve to achieve 90% recall.
#     k_needed = int(np.ceil(recall_target * len(relevant)))
   
#     # Get the top-k distances for 90% recall (excluding itself) (i.e., the distances of the closest `k_needed` neighbors)
#     top_k_distances = valid_distances[:k_needed]

#     # Get the maximum distance for 90% recall (i.e., the distance to the k_needed-th nearest relevant neighbor)
#     if k_needed > 0 and k_needed <= len(valid_distances):
#         max_distance = valid_distances[k_needed - 1]  # k_needed - 1 because of 0-indexing
#     else:
#         max_distance = np.inf  # If there are not enough neighbors to satisfy recall
#     print(max_distance)

#     return max_distance


# # 1. Compute max distance for each query image to achieve 90% recall
# max_distances = []
# for query_index in range(len(train_features)):
#     max_distance = get_max_distance(query_index, recall_target=0.9)
#     max_distances.append(max_distance)

# print(max_distances)
# max_distances = np.array(max_distances)


# print(np.percentile(max_distances,75))

# # Histogram
# plt.hist(max_distances, bins=50, alpha=0.75, color='Teal')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('Aplotsknn/distribution_ft1.pdf')

# # Boxplot
# plt.figure()
# sns.boxplot(max_distances, palette="Set2")
# plt.ylabel('Distance')
# plt.savefig('Aplotsknn/distribution_ft2.pdf')


# plt.figure()
# sns.kdeplot(max_distances, fill=True)
# plt.xlabel('Distance')
# plt.ylabel('Density')
# plt.savefig('Aplotsknn/distribution_ft3.pdf')

# #----------------------------------------------------------------------------------------------------
# # Optimal precision
# #----------------------------------------------------------------------------------------------------

# # 3. Function to get the max distance for 90% precision, excluding `inf` values
# def get_max_distance_for_precision(query_idx, precision_target=0.75):
#     true_label = train_labels[query_idx]  # The true label of the query image
#     # Sort distances and corresponding labels for the query image
#     sorted_idx = np.argsort(distances[query_idx])  # Sort neighbors by increasing distance
#     sorted_distances = distances[query_idx][sorted_idx]  # Sorted distances
#     sorted_labels = train_labels[sorted_idx]  # Corresponding labels of sorted distances


#     # 1. Exclude `inf` values from the sorted distances
#     valid_distances = sorted_distances[sorted_distances != np.inf]
#     valid_labels = sorted_labels[sorted_distances != np.inf]

#     # 2. Iterate through the sorted neighbors and calculate precision
#     relevant_count = 0
#     for i in range(len(valid_distances)):
#         # Check if the current neighbor is relevant (same label as query)
#         if valid_labels[i] == true_label:
#             relevant_count += 1
        
#         # Calculate precision at this point
#         precision = relevant_count / (i + 1)  # (i + 1) is the total number of retrieved items
        
#         # Stop when precision drops below the target precision
#         if precision < precision_target:
#             break
    
#     # 3. Get the maximum distance for the last step where precision was >= target
#     if i > 0:  # Ensure there was at least one valid neighbor retrieved
#         max_distance = valid_distances[i - 1]  # i-1 because we stop after precision drops
#     else:
#         max_distance = np.inf  # If no valid neighbor met the precision target
    
#     return max_distance

# max_distances = []
# for query_index in range(len(train_features)):
#     max_distance = get_max_distance_for_precision(query_index, precision_target=0.6)
#     max_distances.append(max_distance)

# print(max_distances)
# max_distances = np.array(max_distances)
# # Exclude `inf` values
# finite_distances = max_distances[np.isfinite(max_distances)]

# # Histogram
# plt.hist(finite_distances, bins=50, alpha=0.75)
# plt.title('Distribution of Max Distances for 75% Recall')
# plt.xlabel('Max Distance')
# plt.ylabel('Frequency')
# plt.savefig('Aplotsknn/distribution_fixed4.png')

# print(np.percentile(finite_distances, 75))

