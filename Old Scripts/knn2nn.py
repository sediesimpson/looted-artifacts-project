import os
import sys
from tqdm import tqdm
import numpy as np
import heapq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from sklearn.decomposition import PCA

from skimage.io import imread
from skimage.transform import resize
from knndataloader2 import *

from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import time

import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import shuffle

import math
#------------------------------------------------------------------------------------------------------------------------
# Define CustomResNetClassifier
#------------------------------------------------------------------------------------------------------------------------
class CustomResNetExtractor(nn.Module):
    def __init__(self, weights=None):
        super(CustomResNetExtractor, self).__init__()
        self.resnet = models.resnet50(weights=weights)
        self.resnet.requires_grad_(False)  # Freeze all layers

        # Replace the fully connected layer with a custom sequential layer
        self.resnet.fc = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        return x

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
#test_dataset = CustomImageDatasetTest(root_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------------------------------------------------------
# Define the feature extraction function
#------------------------------------------------------------------------------------------------------------------------
def extract_features(dataloader, model, device):
    model.eval()
    features_list = []
    labels_list = []
    img_paths_list = []
    label_counts_list = []
    
    with torch.no_grad():
        for imgs, labels, img_paths, label_counts in tqdm(dataloader, desc="Extracting features"):
            # Move images to the specified device
            imgs = imgs.to(device)
            
            # Extract features
            features = model(imgs)
            #print('feature dimensions:',features.shape)
            
            # Append features and labels to the respective lists
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            img_paths_list.extend(img_paths)
            label_counts_list.extend(label_counts)
 

    # Flatten the list of arrays (handle varying batch sizes)
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    
    return features_array, labels_array, img_paths_list, label_counts_list

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = CustomResNetExtractor(weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_labels, train_img_paths, train_label_counts = extract_features(train_loader, model, device)

# Extract features for test set 
# test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

#------------------------------------------------------------------------------------------------------------------------
# Try out a couple of radii and see what is returned 
#------------------------------------------------------------------------------------------------------------------------
# We set a radius to capture all potential NNs
print(len(train_features))
print('length of train labels:',len(train_labels))

query_index = 50
nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=500).fit(train_features)
distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)
print(len(distances[0]))

# # Get distances and indices for a single query image
# query_index = 50 # Change this to select a different query image
# distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)

# distances = [dist[1:] for dist in distances] # remove first distance as it is itself
# indices = [idx[1:] for idx in indices] # remove first index as it is itself


# print('===========  INDICES ============')
# print(indices)
# print()
# print('=========== DISTANCES ============')
# print(distances)
# print()
# print('=========== TEST IMAGE PATH ============')
# imgs_to_plot = []
# print(train_img_paths[query_index])
# imgs_to_plot.append(train_img_paths[query_index])
# print()
# print('=========== NEIGHBOUR IMAGE PATHS ============')
# for i in indices[0]:
#     imgs_to_plot.append(train_img_paths[i])
# print(imgs_to_plot)

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
# plt.savefig('Aplotsknn/query50notft2.pdf')

# sys.exit()
# #------------------------------------------------------------------------------------------------------------------------
# # Find the pre-processing time for different dataset sizes
# #------------------------------------------------------------------------------------------------------------------------
# save_dir = 'Adatasetsizes/'
# dataset_sizes = []
# preprocessing_times = []
# stddev_preprocessing_times = []

# # Assuming train_features is your full training dataset
# train_features_length = len(train_features)

# # Define different sizes of the dataset to test
# subset_sizes = np.linspace(1, train_features_length, 10, dtype=int)

# # Number of runs to average for each subset size
# num_runs = 5

# for subset_size in subset_sizes:
#     total_preprocessing_time = 0

#     for _ in range(num_runs):
#         # Subset the training dataset
#         subset_indices = np.random.choice(train_features_length, subset_size, replace=False)
#         subset_train_features = train_features[subset_indices]
        
#         # Measure the preprocessing time
#         start_time = time.time()
#         # Pre-processing for kNN approach after optimal radius has been found
#         nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.3).fit(train_features)
#         end_time = time.time()
        
#         preprocessing_time = end_time - start_time
#         total_preprocessing_time += preprocessing_time

#     # Calculate the average preprocessing time for this subset size
#     average_preprocessing_time = total_preprocessing_time / num_runs
#     stddev_preprocessing_time = np.std(total_preprocessing_time)
#     dataset_sizes.append(subset_size)
#     preprocessing_times.append(average_preprocessing_time)
#     stddev_preprocessing_times.append(stddev_preprocessing_time)
    
#     print(f"Subset Size: {subset_size}, Average Preprocessing Time: {average_preprocessing_time}, Std Dev: {stddev_preprocessing_time}")

# # Save arrays to .npy files
# np.save(os.path.join(save_dir, "dataset_sizes_train.npy"), np.array(dataset_sizes))
# np.save(os.path.join(save_dir, "preprocessing_times.npy"), np.array(preprocessing_times))
# np.save(os.path.join(save_dir, "stddev_preprocessing_times.npy"), np.array(stddev_preprocessing_times))
# #------------------------------------------------------------------------------------------------------------------------
# # Plot pre-processing time versus dataset size
# #------------------------------------------------------------------------------------------------------------------------
# # Load the dataset sizes and query times from the .npy files
# dataset_sizes = np.load(os.path.join(save_dir, "dataset_sizes_train.npy"))
# preprocessing_times = np.load(os.path.join(save_dir, "preprocessing_times.npy"))
# stddev_preprocessing_times = np.load(os.path.join(save_dir, "stddev_preprocessing_times.npy"))

# # Print raw data
# print("Dataset Sizes:", dataset_sizes)
# print("Preprocessing Time", preprocessing_times)
# print("Std Dev Preprocessing Time", stddev_preprocessing_times)
# # Plot Query Time vs Dataset Size
# plt.figure(figsize=(10, 5))
# plt.plot(dataset_sizes, preprocessing_times, color='Teal')
# # plt.fill_between(dataset_sizes, 
# #                 preprocessing_times - stddev_preprocessing_times, 
# #                 preprocessing_times + stddev_preprocessing_times , 
# #                 color='b', alpha=0.2, label='+- 1 Standard Deviation')
# plt.xlabel('Dataset Size')
# plt.ylabel('Pre-processing time')
# plt.title('Preprocessing Time vs Dataset Size')
# plt.grid(True)
# plt.savefig('Adatasetsizes/preprocessing.png')


# sys.exit()
#------------------------------------------------------------------------------------------------------------------------
# Find the optimal radius
#------------------------------------------------------------------------------------------------------------------------
allcurves = []

for query_index, query in enumerate(train_features):
    correct = 0 
    curve = []
    num_correct = train_label_counts[query_index]


    tf = train_features[query_index]
    nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=500).fit(train_features)
    distances, indices = nbrs.radius_neighbors([query], sort_results=True)

    distances = distances[0] # remove first distance as it is itself
    print(f'nearest neighbor={distances[0]}')
    indices = indices[0] # remove first index as it is itself

    true_label = train_labels[query_index]

    for i, idx in enumerate(indices, start=1):
        neighbor_label = train_labels[idx]
        if neighbor_label == true_label:
            correct += 1
            print(f'neighbor_label={neighbor_label}, true label={true_label}')
        if i == 1:
            print(f'index={idx}')
        
        precision = correct / i 
        recall = correct / num_correct.item()

        if recall > 1:
            print(f'recall={recall}, correct={correct}, num correct={num_correct}, num correct item={num_correct.item()}')

        #print(f'precision={precision}, recall={recall}')

        curve.append([i, precision, recall])

    if query_index == 3:
        break

    allcurves.append(curve)

allcurves_stacked = np.stack(allcurves, axis=0)

# print(allcurves_stacked)
print('shape of curve:', allcurves_stacked.shape)
print('indices:', allcurves_stacked[0][:,0])

average_precision_at_k = []
average_recall_at_k = []
for i in range(len(allcurves_stacked[0][:,0])):
    average_precision = np.mean(allcurves_stacked[:,i,1])
    average_recall = np.mean(allcurves_stacked[:,i,2])
    average_precision_at_k.append(average_precision)
    average_recall_at_k.append(average_recall)    

print('average precision:', average_precision_at_k)
print('average recall:', average_recall_at_k)
x_values = range(0,776)


plt.figure(figsize=(12, 4))
plt.plot(x_values, average_precision_at_k, label='Precision', color='Teal')
plt.plot(x_values, average_recall_at_k, label='Recall', color='rebeccapurple')
plt.xlabel("Number of neighbours")
plt.legend()
plt.savefig('Aplotsknn/precisionrecallNN.pdf')




plt.figure(figsize=(12, 4))
plt.plot(average_recall_at_k, average_precision_at_k, color='Teal')
plt.xlabel("Recall")
plt.ylabel('Precision')
plt.savefig('Aplotsknn/prcurve.pdf')



sys.exit()
print(allcurves_stacked[0])

plt.figure(figsize=(12, 4))
plt.plot(allcurves_stacked[0][:, 0], allcurves_stacked[0][:, 1])
plt.title("Precision")
plt.xlabel("Number of neighbours")
plt.ylabel("Precision")
plt.savefig('Aplotsknn/precision.png')


plt.figure(figsize=(12, 4))
plt.plot(allcurves_stacked[0][:, 0], allcurves_stacked[0][:, 2])
plt.title("Recall")
plt.xlabel("Number of neighbours")
plt.ylabel("Recall")
plt.savefig('Aplotsknn/recall.png')

plt.figure(figsize=(12, 4))
plt.plot(allcurves_stacked[0][:, 2], allcurves_stacked[0][:, 1])
plt.title("PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig('Aplotsknn/prcurve.png')

sys.exit()


        

