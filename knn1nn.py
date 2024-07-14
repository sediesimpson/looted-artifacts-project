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

#------------------------------------------------------------------------------------------------------------------------
# Plot nearest neighbours
#------------------------------------------------------------------------------------------------------------------------
def save_combined_images(test_imgs, train_imgs, indices, output_dir='output'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for query_idx in range(len(test_imgs)):
        query_img = test_imgs[query_idx]
        neighbor_indices = indices[query_idx]  # Nearest neighbor indices
        neighbors = [train_imgs[idx] for idx in neighbor_indices]
        
        # Determine the width of the combined image
        total_width = query_img.width + sum(neighbor.width for neighbor in neighbors)
        max_height = max(query_img.height, max(neighbor.height for neighbor in neighbors))
        
        # Combine the images side by side
        combined_img = Image.new('RGB', (total_width, max_height))
        combined_img.paste(query_img, (0, 0))
        
        current_width = query_img.width
        for neighbor in neighbors:
            combined_img.paste(neighbor, (current_width, 0))
            current_width += neighbor.width
        
        # Save the combined image
        combined_img.save(f"{output_dir}/query_neighbor_{query_idx + 1}.png")
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
    
#------------------------------------------------------------------------------------------------------------------------
# Split the dataset 
#------------------------------------------------------------------------------------------------------------------------
# Function to ensure no data leakage
def split_dataset(dataset, test_split=0.2, val_split=0.1, shuffle=True, random_seed=42):
    # Identify unique images by their basename
    unique_images = list(set(os.path.basename(path) for path in dataset.img_paths))
    
    # Split the dataset based on unique images
    train_val_imgs, test_imgs = train_test_split(unique_images, test_size=test_split, random_state=random_seed)
    train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_split/(1-test_split), random_state=random_seed)
    
    train_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in train_imgs]
    val_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in val_imgs]
    test_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in test_imgs]
    
    return train_indices, val_indices, test_indices
#---------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
batch_size = 32
random_seed = 42

# Create train dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/duplicatedata"
dataset = CustomImageDatasetDup(root_dir)

# Split dataset without data leakage
train_indices, val_indices, test_indices = split_dataset(dataset, test_split=0.15, val_split=0.15, shuffle=True, random_seed=42)

# Create data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

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
#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = dataset.count_unique_labels()
hidden_features = 512
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)
val_features, val_labels, val_img_paths = extract_features(val_loader, model, device)
test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=100)
train_features = pca.fit_transform(train_features,)
val_features = pca.transform(val_features)
test_features = pca.transform(test_features)

#------------------------------------------------------------------------------------------------------------------------
# Determine optimal radius
#------------------------------------------------------------------------------------------------------------------------
radii = np.linspace(0.01, 1, 100)  # Define a range of radii to test

best_radius = None
best_precision = -1

for radius in radii:
    nbrs = NearestNeighbors(metric='cosine', radius=radius, algorithm='brute').fit(train_features)
    distances, indices = nbrs.radius_neighbors(val_features)

    true_positives = 0
    false_positives = 0

    for i, neighs in enumerate(indices):
        if len(neighs) == 0:
            continue  # Handle case where no neighbors are found
        neigh_labels = train_labels[neighs]
        query_label = val_labels[i]
        matching_neighbors = neigh_labels == query_label
        
        true_positives += np.sum(matching_neighbors)
        false_positives += np.sum(~matching_neighbors)

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0

    if precision > best_precision:
        best_precision = precision
        best_radius = radius

print(f"Optimal radius on validation set: {best_radius} with precision: {best_precision}")


#------------
# test set
#--------------

# Use the best_radius found from the validation set
nbrs = NearestNeighbors(metric='cosine', radius=best_radius, algorithm='brute').fit(train_features)
distances, indices = nbrs.radius_neighbors(test_features)

true_positives = 0
false_positives = 0
total_predictions = 0
precision_scores = []

# Store test image paths and their nearest neighbors
test_img_neighbors = []

for i, neighs in enumerate(indices):
    test_img_path = test_img_paths[i]
    neighbors = []
    
    if len(neighs) == 0:
        test_img_neighbors.append((test_img_path, neighbors))
        continue  # Handle case where no neighbors are found
    
    neigh_labels = train_labels[neighs]
    query_label = test_labels[i]
    matching_neighbors = neigh_labels == query_label
    
    for j, match in enumerate(matching_neighbors):
        neighbor_path = train_img_paths[neighs[j]]
        neighbors.append((neighbor_path, match))
        
        if match:
            true_positives += 1
        else:
            false_positives += 1

    total_predictions += len(neigh_labels)
    test_img_neighbors.append((test_img_path, neighbors))

if (true_positives + false_positives) > 0:
    test_precision = true_positives / (true_positives + false_positives)
else:
    test_precision = 0

print(f"Test precision with optimal radius: {test_precision}")

# Print test image paths and their nearest neighbors
for test_img, neighbors in test_img_neighbors:
    print(f"Test Image: {test_img}")
    for neighbor_path, match in neighbors:
        print(f"    Neighbor: {neighbor_path}, Match: {match}")

sys.exit()

# Function to load and display images
def load_and_display_image(img_path, ax, title):
    try:
        image = Image.open(img_path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        ax.axis('off')

# Select 5 random indices from the test set
num_samples_to_visualize = 5
random_indices = np.random.choice(len(test_features), num_samples_to_visualize, replace=False)


# Display the random test samples along with their nearest neighbors
for row, idx in enumerate(random_indices):
    # Construct the full path for the test image
    test_image_path = os.path.join(root_dir, test_img_paths[idx])
    actual_label = test_labels[idx]  # Assume y_test contains the actual labels
    distances_to_neighbors = distances[idx]
    neighbor_indices = indices[idx]

    # Debugging prints
    print(f"Test image: {test_image_path}")
    print(f"Neighbors: {neighbor_indices}")
    print(f"Neighbor distances: {distances_to_neighbors}")

    if len(neighbor_indices) == 0:
        print(f"No neighbors found for test image {idx}")
        continue

    # Number of neighbors for the current test image
    num_neighbors = len(neighbor_indices)

    # Create a new figure for each test image
    fig, axes = plt.subplots(1, num_neighbors + 1, figsize=(15, 5))  # 1 row, num_neighbors+1 columns

    # Display the test image
    load_and_display_image(test_image_path, axes[0], f'Test Image\nActual: {actual_label}')

    # Display the nearest neighbors
    for col, neighbor_idx in enumerate(neighbor_indices):
        # Construct the full path for the neighbor image
        neighbor_path = os.path.join(root_dir, train_img_paths[neighbor_idx])
        neighbor_label = train_labels[neighbor_idx]  # Neighbor labels
        distance = distances_to_neighbors[col]
        load_and_display_image(neighbor_path, axes[col + 1], f'Neighbor {col + 1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')

    plt.tight_layout()
    plt.savefig(f'neighbors_visualization_pca{row}.png')














# # Measure the preprocessing time
# start_time = time.time()
# # Train the KNeighborsClassifier
# nbrs = NearestNeighbors(radius=0.006, metric='cosine', algorithm='brute').fit(train_features)
# end_time  = time.time()
# preprocess_time = end_time-start_time
# print(f'Pre-processing time: {preprocess_time:.2f}')

# start_time_query = time.time()
# # Find the nearest neighbors for each point in the query dataset
# distances, indices = nbrs.kneighbors(test_features)

# end_time_query  = time.time()
# query_time = end_time_query-start_time_query
# print(f'Query time: {query_time:.2f}')







sys.exit()

# Calculate precision, recall, F1
n_neighbors = 2
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


# Calculate memory usage
data_memory = sys.getsizeof(train_features)
model_memory = sys.getsizeof(nbrs)
indices_memory = sys.getsizeof(nbrs._fit_X)  # Memory used by the training data in the model
distances_memory = sys.getsizeof(nbrs._tree)  # Memory used by the tree if applicable

total_memory = data_memory + model_memory + indices_memory + distances_memory

print(f'Data memory usage: {data_memory} bytes')
print(f'Model memory usage: {model_memory} bytes')
print(f'Indices memory usage: {indices_memory} bytes')
print(f'Distances memory usage: {distances_memory} bytes')
print(f'Total memory usage: {total_memory} bytes')

# Calculating the number of comparisons done
N = len(train_features)
Q = len(test_features)
# Total comparisons for brute force
total_comparisons = N * Q

# Call the function to plot the query images and their nearest neighbors
save_combined_images(test_img_paths, train_img_paths, indices, output_dir='output2NN')