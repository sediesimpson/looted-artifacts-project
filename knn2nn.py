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
# num_classes = train_dataset.count_unique_labels()
# hidden_dim = 256  # Intermediate hidden layer size
# weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
# model = CustomResNetClassifier(hidden_dim, num_classes, weights=weights)
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = CustomResNetExtractor(weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_labels, train_img_paths, train_label_counts = extract_features(train_loader, model, device)

# Extract features for test set 
# test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

#------------------------------------------------------------------------------------------------------------------------
# Find the optimal distance threshold for the knn method, using precision
#------------------------------------------------------------------------------------------------------------------------
# Range of thresholds to evaluate
thresholds = np.linspace(0.01, 0.1, 100)

# We set a large radius to capture all potential NNs
nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=1.0).fit(train_features)

# Get distances and indices for a single query image
# query_index = 8  # Change this to select a different query image
# distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)

# Exclude the point itself if it appears in its own neighbors
# for i, (dist, idx) in enumerate(zip(distances, indices)):
#     # Create a mask to exclude the point itself
#     mask = (dist != 0)  # Assuming the distance of the point to itself is exactly 0
#     # Apply the mask to filter distances and indices
#     distances[i] = dist[mask]
#     indices[i] = idx[mask]

# distances = [dist[1:] for dist in distances] # remove first distance as it is itself
# indices = [idx[1:] for idx in indices] # remove first index as it is itself

# print('distances:', distances)
# #print('indices:', indices)

# filtered_indices = [index for dist, index in zip(distances[0], indices[0]) if dist < 0.055]
# filtered_distances = [dist for dist, index in zip(distances[0], indices[0]) if dist < 0.055]

# print('=========== FILTERED INDICES ============')
# print(filtered_indices)
# print()
# print('=========== FILTERED DISTANCES ============')
# print(filtered_distances)
# print()
# print('=========== TEST IMAGE PATH ============')
# print(train_img_paths[query_index])
# print()
# print('=========== NEIGHBOUR IMAGE PATHS ============')
# for i in filtered_indices:
#     print(train_img_paths[i])

#------------------------------------------------------------------------------------------------------------------------
# Find the optimal radius
#------------------------------------------------------------------------------------------------------------------------
radii = np.linspace(0.05, 0.5, 100)

best_precision = 0
best_radius = 0 
best_recall = 0

recalls = []

for radius in radii:
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0 
    
    nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=radius).fit(train_features)

    for query_index in range(len(train_features)):
        distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)

        distances = distances[0][1:] # remove first distance as it is itself
        indices = indices[0][1:] # remove first index as it is itself

        true_label = train_labels[query_index]
        neighbor_labels = train_labels[indices]

        if neighbor_labels.size > 0:
            for label in neighbor_labels:
                if label == true_label:
                    true_positives +=1 
                else:
                    true_negatives +=1 

            fn = train_label_counts[query_index] - true_positives

            if fn < 0:
                fn = 0

            false_negatives += fn

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0

    recalls.append(recall)
    if recall >= best_recall:
        best_recall = recall
        best_radius = radius

        
print(f"Best radius: {best_radius} with recall: {best_recall}")


        





sys.exit()
#------------------------------------------------------------------------------------------------------------------------
# Find the optimal distance threshold for the knn method, using precision
#------------------------------------------------------------------------------------------------------------------------

# Range of thresholds to evaluate
thresholds = np.linspace(0.01, 0.1, 100)

# We set a large radius to capture all potential NNs
nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=1.0).fit(train_features)

best_precision = 0
best_threshold = 0 
best_recall = 0

precisions = []
recalls = []
for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0 
        
        for query_index in range(len(train_features)):
            distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)
            print('distances:', distances)

            distances = distances[0][1:] # remove first distance as it is itself
            indices = indices[0][1:] # remove first index as it is itself

            filtered_indices = [index for dist, index in zip(distances, indices) if dist < threshold]

            if filtered_indices:
                # For simplicity, use the most common label among neighbors as the prediction
                neighbor_labels = train_labels[filtered_indices]
                predicted_label = np.bincount(neighbor_labels).argmax()

                true_label = train_labels[query_index]

                if predicted_label == true_label:
                    if predicted_label == 1:  # Assuming 1 is the positive class
                        true_positives += 1
                    else:
                     true_negatives += 1

                else:
                    if predicted_label == 1:  # Predicted positive, but actually negative
                        false_positives += 1
                    else:  # Predicted negative, but actually positive
                        false_negatives += 1


        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        precisions.append(precision)
        if precision >= best_precision:
            best_precision = precision
            best_threshold = threshold

        # recalls.append(recall)
        # if recall >= best_recall:
        #     best_recall = recall
        #     best_threshold = threshold

print(f"Best threshold: {best_threshold} with precision: {best_precision}")


        

