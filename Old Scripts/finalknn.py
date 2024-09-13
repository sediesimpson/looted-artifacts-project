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
from sklearn.metrics import precision_score, recall_score, f1_score, pairwise_distances
from sklearn.utils import shuffle

import math
import seaborn as sns
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
test_dataset = CustomImageDatasetTest(root_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
test_features, test_labels, test_img_paths, test_label_counts = extract_features(test_loader, model, device)

# ------------------------------------------------------------------------------------------------------------------------
# Running the knn
# ------------------------------------------------------------------------------------------------------------------------

# query_index = 88
# nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.14).fit(test_features)
# distances, indices = nbrs.radius_neighbors([test_features[query_index]], sort_results=True)
# print(len(distances[0]))

# print('===========  INDICES ============')
# print(indices)
# print()
# print('=========== DISTANCES ============')
# print(distances)
# print()
# print('=========== TEST IMAGE PATH ============')
# imgs_to_plot = []
# print(test_img_paths[query_index])
# print()
# print('=========== NEIGHBOUR IMAGE PATHS ============')
# for i in indices[0]:
#     imgs_to_plot.append(test_img_paths[i])
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
# plt.savefig('Aplotsknn/query88NOTFT.pdf')

# #------------------------------------------------------------------------------------------------------------------------
# # Find the optimal radius
# #------------------------------------------------------------------------------------------------------------------------
# allcurves = []
# target_recall = 0.9
# distribution = []

# for query_index, query in enumerate(train_features):
#     correct = 0 
#     curve = []
#     num_correct = train_label_counts[query_index]

#     tf = train_features[query_index]
#     nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=500).fit(train_features)
#     distances, indices = nbrs.radius_neighbors([query], sort_results=True)

#     distances = distances[0] # remove first distance as it is itself
#     print(f'nearest neighbor={distances[0]}')
#     indices = indices[0] # remove first index as it is itself

#     true_label = train_labels[query_index]

#     for i, (dist,idx) in enumerate(zip(distances, indices), start=0):
#         neighbor_label = train_labels[idx]
#         if neighbor_label == true_label:
#             correct += 1
#             print(f'neighbor_label={neighbor_label}, true label={true_label}')

#         precision = correct / (i + 1)
#         recall = correct / num_correct.item()

#         if recall >= target_recall:
#             distribution.append(distances[i])
#             break

# print('Distribution:', distribution)
# print('Distribution:', len(distribution))

# # plt.hist(distribution, color = 'mediumseagreen', density=True)
# # plt.xlabel('Radius')
# # plt.ylabel('Density')

# # Overlay a KDE line

# fig, ax = plt.subplots()
# dist_plot = sns.kdeplot(distribution, color='teal')

# ax.set_xlabel('Radius')
# ax.set_ylabel('Density')

# lines_for_plot = dist_plot.get_lines()
# print('lines for plot:',lines_for_plot)

# for line in lines_for_plot:
#     # ax.plot(line.get_data())
#     x, y = line.get_data()
#     print(x[np.argmax(y)])
#     ax.axvline(x[np.argmax(y)], ls='--', color='black')

# plt.savefig('Aplotsknn/distribution.pdf')

# #----------------------------------------------------------------------------------------------------------------------------
# # Figuring out query time
# #----------------------------------------------------------------------------------------------------------------------------
# save_dir = 'Adatasetsizes/'
# # Initialize lists to store dataset sizes and query times
# dataset_sizes = []
# query_times = []


# subset_sizes1 = range(10,100,10)
# subset_sizes2 = range(100,1000,100)
# subset_sizes3 = range(1000,3500,500)
# subset_sizes = np.concatenate((subset_sizes1, subset_sizes2, subset_sizes3))
# # subset_sizes = range(100,1000,100)

# # Number of queries to average for each subset size
# num_queries = 10

# nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.13).fit(train_features)

# # Iterate through each subset size
# for subset_size in subset_sizes:
#     # Subset the test dataset
#     subset_indices = np.random.choice(len(train_features), subset_size, replace=False)
#     subset_train_features = train_features[subset_indices]


#       # Measure the query times for multiple queries
#     total_query_time = 0
#     for _ in range(num_queries):

#         start_time = time.time()

#         for query_index in range(len(subset_train_features)):
#             # Find all the neighbors
#             distances, indices = nbrs.radius_neighbors([subset_train_features[query_index]], sort_results=True)

#         # End time for the query
#         end_time = time.time()
#         query_time = end_time - start_time

#         # Accumulate the query time
#         total_query_time += query_time

#     # Calculate the average query time for this subset size
#     average_query_time = total_query_time / num_queries
#     dataset_sizes.append(subset_size)
#     query_times.append(average_query_time)

#     print(f"Subset Size: {subset_size}, Average Query Time: {average_query_time}")

# # Optionally, you can save the dataset_sizes and query_times for further analysis
# np.save(os.path.join(save_dir, "dataset_sizes_KNN_NOFT_FULL.npy"), np.array(dataset_sizes))
# np.save(os.path.join(save_dir, "query_times_KNN_NOFT_FULL.npy"), np.array(query_times))


# #----------------------------------------------------------------------------------------------------------------------------
# # Figuring out query time 2
# #----------------------------------------------------------------------------------------------------------------------------
# save_dir = 'Adatasetsizes/'
# # Initialize lists to store dataset sizes and query times
# dataset_sizes = []
# query_times = []

# subset_sizes1 = range(10,100,10)
# subset_sizes2 = range(100,1000,100)
# subset_sizes3 = range(1000,3500,500)
# subset_sizes = np.concatenate((subset_sizes1, subset_sizes2, subset_sizes3))


# # Number of queries to average for each subset size
# num_queries = 10

# # Iterate through each subset size
# for subset_size in subset_sizes:
#     # Subset the test dataset
#     subset_indices = np.random.choice(len(train_features), subset_size, replace=False)
#     subset_train_features = train_features[subset_indices]

#     nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.13).fit(subset_train_features)
    

#     # Measure the query times for multiple queries
#     total_query_time = 0
#     for _ in range(num_queries):
        
#         query_index = np.random.choice(len(subset_train_features))

#         start_time = time.time()

#         # distances = pairwise_distances(subset_train_features[query_index].reshape(1,-1), subset_train_features, metric='cosine')
#         # print(distances)
#         distances, indices = nbrs.radius_neighbors([subset_train_features[query_index]], sort_results=True)
#         end_time = time.time()

#         query_time = end_time - start_time
#         total_query_time += query_time

#     # Calculate the average query time for this subset size
#     average_query_time = total_query_time / num_queries
#     dataset_sizes.append(subset_size)
#     query_times.append(average_query_time)

#     print(f"Subset Size: {subset_size}, Average Query Time: {average_query_time}")

# # save the dataset_sizes and query_times for further analysis
# np.save(os.path.join(save_dir, "dataset_sizes_KNN_NOFT_FULL.npy"), np.array(dataset_sizes))
# np.save(os.path.join(save_dir, "query_times_KNN_NOFT_FULL.npy"), np.array(query_times))

#----------------------------------------------------------------------------------------------------------------------------
# Testing the precision and recall 
#----------------------------------------------------------------------------------------------------------------------------
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0 
precision = 0
recall = 0

nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.5).fit(test_features)

for query_index in range(len(test_features)):
    distances, indices = nbrs.radius_neighbors([test_features[query_index]], sort_results=True)

    distances = distances[0][1:] # remove first distance as it is itself
    indices = indices[0][1:] # remove first index as it is itself

    true_label = test_labels[query_index]
    neighbor_labels = test_labels[indices]

    if neighbor_labels.size > 0:
        for label in neighbor_labels:
            if label == true_label:
                true_positives +=1 

        fn = test_label_counts[query_index] - true_positives

        if fn < 0:
            fn = 0

        false_negatives += fn

        tn = len(test_features) - false_negatives

        if tn < 0: 
            tn = 0 

        true_negatives += tn

        fp = len(neighbor_labels) - true_positives

        if fp < 0:
            fp = 0

        false_positives += fp


if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0

if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives) 
else:
    precision = 0

print(f'True positives:{true_positives}, false positives:{false_positives}, true negatives: {true_negatives}, false negatives = {false_negatives}, recall: {recall}, precision:{precision}')

