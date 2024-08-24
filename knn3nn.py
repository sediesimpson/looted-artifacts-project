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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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


# Extract features for training and validation sets
train_features, train_labels, train_img_paths, train_label_counts = extract_features(train_loader, model, device)

# ------------------------------------------------------------------------------------------------------------------------
# Try out a couple of radii and see what is returned 
# ------------------------------------------------------------------------------------------------------------------------
# We set a radius to capture all potential NNs
nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.004).fit(train_features)

# Get distances and indices for a single query image
query_index = 10 # Change this to select a different query image
distances, indices = nbrs.radius_neighbors([train_features[query_index]], sort_results=True)

distances = [dist[1:] for dist in distances] # remove first distance as it is itself
indices = [idx[1:] for idx in indices] # remove first index as it is itself

print('train features length:',len(train_features))
print('===========  INDICES ============')
print(len(indices[0]))
print()
print('=========== DISTANCES ============')
print(len(distances[0]))
print()
print('=========== TEST IMAGE PATH ============')
imgs_to_plot = []
print(train_img_paths[query_index])
imgs_to_plot.append(train_img_paths[query_index])
print()
print('=========== NEIGHBOUR IMAGE PATHS ============')
for i in indices[0]:
    imgs_to_plot.append(train_img_paths[i])
print(imgs_to_plot)

print('=========== PLOTTING IMAGES ============')
# Number of images
num_images = len(imgs_to_plot)

# Maximum number of columns
max_cols = 5
rows = math.ceil(num_images / max_cols)  # Calculate the number of rows needed

# Create a figure with a dynamic grid of subplots (multiple rows if needed)
fig, axes = plt.subplots(rows, min(num_images, max_cols), figsize=(min(num_images, max_cols) * 5, rows * 5))  # Adjust figsize as needed

# Flatten axes array for easier iteration if grid has more than 1 row
axes = axes.flatten() if num_images > 1 else [axes]

# Loop through each image path and plot them
for i, img_path in enumerate(imgs_to_plot):
    img = Image.open(img_path)  # Open the image
    axes[i].imshow(img)  # Display the image
    axes[i].axis('off')  # Hide the axes

    # Add labels
    if i == 0:
        axes[i].set_title("Query Image")
    else:
        axes[i].set_title(f"Neighbour {i}")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Show the plot with all images
plt.savefig('Aplotsknn/query10ft.png')

sys.exit()
#------------------------------------------------------------------------------------------------------------------------
# Find the optimal radius
#------------------------------------------------------------------------------------------------------------------------
radii = np.linspace(0.05, 0.5, 100)


best_radius = 0 
best_recall = 0
final_precision = 0
final_f1_score = 0 

recalls = []
precisions = []
f1_scores = []

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

    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0 

    recalls.append(recall)
    precisions.append(precision)
    f1_scores.append(f1)

    if recall >= best_recall:
        best_recall = recall
        best_radius = radius
        final_precision = precision
        final_f1_score = f1

print(f"Best radius: {best_radius} with best recall: {best_recall}, precision:{final_precision} and f1_score:{final_f1_score}")
