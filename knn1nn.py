import os
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from multidataloader import *
from knndataloader import *

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
#------------------------------------------------------------------------------------------------------------------------
# Define the CustomClassifier module
#------------------------------------------------------------------------------------------------------------------------
class CustomClassifier(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#------------------------------------------------------------------------------------------------------------------------
# Define the CustomResnet50 module
#------------------------------------------------------------------------------------------------------------------------
class CustomResNet50(nn.Module):
    def __init__(self, num_classes, hidden_features):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze the convolutional layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        num_features = self.resnet50.fc.in_features  # This should be 2048 for ResNet50
        self.resnet50.fc = nn.Identity()  # Remove the existing fully connected layer
        self.custom_classifier = CustomClassifier(num_features, hidden_features, num_classes)

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.resnet50.avgpool(self.resnet50.layer4(self.resnet50.layer3(self.resnet50.layer2(self.resnet50.layer1(self.resnet50.maxpool(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x)))))))))
        x = torch.flatten(x, 1)
        x = self.custom_classifier(x)
        return x
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
print("1 Nearest Neighbour KNN Search")
batch_size = 32
random_seed = 42

# Create train dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"
dataset = CustomImageDataset2(root_dir)

# Get label information
# label_info = dataset.get_label_info()
# print("Label Information for Training Data:", label_info)

# # Get the number of images per label
# label_counts = dataset.count_images_per_label()
# print("Number of images per label for Training Data:", label_counts)

train_loader = DataLoader(dataset, batch_size=batch_size)
#--------------------------------------------------------------------------------------------------------------------------
# Get Test Dataset 
#--------------------------------------------------------------------------------------------------------------------------

root_dir_test = "/rds/user/sms227/hpc-work/dissertation/data/1NN"
test_dataset = CustomImageDataset3(root_dir_test)
test_loader =  DataLoader(test_dataset, batch_size=batch_size)
#------------------------------------------------------------------------------------------------------------------------
# Define the feature extraction function
#------------------------------------------------------------------------------------------------------------------------
def extract_features(dataloader, model, device):
    model.eval()
    features_list = []
    labels_list = []
    img_paths_list = []
    
    with torch.no_grad():
        for imgs, labels, img_paths in tqdm(dataloader, desc="Extracting features"):
            # Move images to the specified device
            imgs = imgs.to(device)
            
            # Extract features
            features = model.resnet50.avgpool(
                model.resnet50.layer4(
                    model.resnet50.layer3(
                        model.resnet50.layer2(
                            model.resnet50.layer1(
                                model.resnet50.maxpool(
                                    model.resnet50.relu(
                                        model.resnet50.bn1(
                                            model.resnet50.conv1(imgs)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            features = torch.flatten(features, 1)
            
            # Append features and labels to the respective lists
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            img_paths_list.extend(img_paths)

    # Flatten the list of arrays (handle varying batch sizes)
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    return features_array, labels_array, img_paths_list

def extract_features_nolabel(dataloader, model, device):
    model.eval()
    features_list = []
    img_paths_list = []
    
    with torch.no_grad():
        for imgs, img_paths in tqdm(dataloader, desc="Extracting features"):
            # Move images to the specified device
            imgs = imgs.to(device)
            
            # Extract features
            features = model.resnet50.avgpool(
                model.resnet50.layer4(
                    model.resnet50.layer3(
                        model.resnet50.layer2(
                            model.resnet50.layer1(
                                model.resnet50.maxpool(
                                    model.resnet50.relu(
                                        model.resnet50.bn1(
                                            model.resnet50.conv1(imgs)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            features = torch.flatten(features, 1)
            
            # Append features and labels to the respective lists
            features_list.append(features.cpu().numpy())
            img_paths_list.extend(img_paths)

    # Flatten the list of arrays (handle varying batch sizes)
    features_array = np.concatenate(features_list, axis=0)
    
    return features_array, img_paths_list
#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 28
hidden_features = 512
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)
test_features, test_img_paths = extract_features_nolabel(test_loader, model, device)

# Measure the preprocessing time
start_time = time.time()
# Train the KNeighborsClassifier
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(train_features)
end_time  = time.time()
preprocess_time = end_time-start_time
print(f'Pre-processing time: {preprocess_time:.2f}')

start_time_query = time.time()
# Find the nearest neighbors for each point in the query dataset
distances, indices = nbrs.kneighbors(test_features)

end_time_query  = time.time()
query_time = end_time_query-start_time_query
print(f'Query time: {query_time:.2f}')

# Calculate precision, recall, F1
n_neighbors = 1
true_positives = 0
false_positives = 0
true_negatives = 0 
false_negatives = 0 
total_predictions = 0

for i, neighbors in enumerate(indices):
    query_filename = os.path.basename(test_img_paths[i])
    neighbor_filenames = [os.path.basename(train_img_paths[idx]) for idx in neighbors]
    
    print(query_filename)
    print(neighbor_filenames)

    # Check if query_filename matches any of the neighbor filenames
    if query_filename in neighbor_filenames:
        true_positives += 1
    else:
        false_negatives += 1  # Query filename not found in nearest neighbors
    false_positives += n_neighbors - neighbor_filenames.count(query_filename)
    total_predictions += n_neighbors

# Calculate precision, recall
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')

