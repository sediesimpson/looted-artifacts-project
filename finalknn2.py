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
from sklearn.metrics.pairwise import cosine_distances
import math
import seaborn as sns
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
torch.manual_seed(50)
torch.cuda.manual_seed(50)
torch.cuda.manual_seed_all(50)

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

#------------------------------------------------------------------------------------------------------------------------
# Find the optimal radius
#------------------------------------------------------------------------------------------------------------------------
allcurves = []
target_recall = 0.9
distribution = []

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

    for i, (dist,idx) in enumerate(zip(distances, indices), start=0):
        neighbor_label = train_labels[idx]
        if neighbor_label == true_label:
            correct += 1
            print(f'neighbor_label={neighbor_label}, true label={true_label}')

        precision = correct / (i + 1)
        recall = correct / num_correct.item()

        if recall >= target_recall:
            distribution.append(distances[i])
            break

print('Distribution:', distribution)
print('Distribution:', len(distribution))

# plt.hist(distribution)

# Overlay a KDE line

fig, ax = plt.subplots()
dist_plot = sns.kdeplot(distribution, color='r', linewidth=2)

lines_for_plot = dist_plot.get_lines()
print('lines for plot:',lines_for_plot)

for line in lines_for_plot:
    # ax.plot(line.get_data())
    x, y = line.get_data()
    print(x[np.argmax(y)])
    ax.axvline(x[np.argmax(y)], ls='--', color='black')


plt.savefig('Aplotsknn/distribution2.png')



