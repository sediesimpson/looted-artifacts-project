# Take average of ~10 executions
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import os 
import sys
from multidataloader import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations
from copy import copy
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
from knndataloader2 import * 
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
torch.manual_seed(51)
torch.cuda.manual_seed(51)
torch.cuda.manual_seed_all(51)

# Create train dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/duplicatedata"
train_dataset = CustomImageDatasetTrain(root_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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

#----------------------------------------------------------------------------------------------------------------------------
# Define LSH CLASS
#----------------------------------------------------------------------------------------------------------------------------
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

class LSH:
    def __init__(self, data):
        self.data = data
        self.model = None

    def __generate_random_vectors(self, num_vector, dim):
        return np.random.randn(dim, num_vector)

    def train(self, num_vector, seed=None):
        dim = self.data.shape[1]
        if seed is not None:
            np.random.seed(seed)

        random_vectors = self.__generate_random_vectors(num_vector, dim)
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        table = {}

        # Partition data points into bins
        bin_index_bits = (self.data.dot(random_vectors) >= 0)

        # Encode bin index bits into integers
        bin_indices = bin_index_bits.dot(powers_of_two)

        # Update `table` so that `table[i]` is the list of image ids with bin index equal to i.
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                table[bin_index] = []
            table[bin_index].append(data_index)

        self.model = {'bin_indices': bin_indices, 'table': table, 'random_vectors': random_vectors, 'num_vector': num_vector}
        return self

    def __search_nearby_bins(self, query_bin_bits, table, search_radius=2, initial_candidates=set()):
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        candidate_set = copy(initial_candidates)

        for different_bits in combinations(range(num_vector), search_radius):
            alternate_bits = copy(query_bin_bits)
            for i in different_bits:
                alternate_bits[i] = 1 if alternate_bits[i] == 0 else 0

            nearby_bin = alternate_bits.dot(powers_of_two)

            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])

        return candidate_set

    def query(self, query_vec, max_search_radius, initial_candidates=set()):
        if not self.model:
            raise ValueError('Model not yet built. Exiting!')

        data = self.data
        table = self.model['table']
        random_vectors = self.model['random_vectors']

        bin_index_bits = (query_vec.dot(random_vectors) >= 0).flatten()

        candidate_set = set()
        for search_radius in range(max_search_radius + 1):
            candidate_set = self.__search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=initial_candidates)

        # Ensure the indices are integers before using them
        candidate_set = set(map(int, candidate_set))

        # Handle the case where no candidates are found
        if not candidate_set:
            return DataFrame({'id': [], 'distance': []})

        candidates = data[np.array(list(candidate_set), dtype=int), :]
        
        nearest_neighbors = DataFrame({'id': list(candidate_set)})

        # Reshape query_vec to be a 2D array
        query_vec = query_vec.reshape(1, -1)
        
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec, metric='cosine').flatten()

        # Sort by distance
        nearest_neighbors = nearest_neighbors.sort_values(by='distance')

        return nearest_neighbors



# Train LSH on the full training dataset
lsh = LSH(train_features)
lsh.train(num_vector=10, seed=42)

# Find the nearest neighbours
query_index = 50
query_vec = train_features[query_index]
all_neighbors = lsh.query(query_vec, 0)
print('==============true label==================')
true_label = train_labels[query_index]
print('true label:', true_label)

imgs_to_plot = []
imgs_to_plot.append(train_img_paths[query_index])
print('==============neigbour label==================')
neighbor_labels = [train_labels[i] for i in all_neighbors['id']]
print('neighbor labels:',neighbor_labels)

neighbor_paths = [train_img_paths[i] for i in neighbor_labels]
imgs_to_plot.extend(neighbor_paths)
print('images to plot:', imgs_to_plot)
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
plt.savefig('Bplotslsh/query50noft.png')

sys.exit()

# #----------------------------------------------------------------------------------------------------------------------------
# Find best search radius 
#----------------------------------------------------------------------------------------------------------------------------
radii = range(0,10)

recalls = []
precisions = []
f1_scores = []
neighbor_labels = []

best_radius = 0 
best_recall = 0
final_precision = 0
final_f1_score = 0 

for radius in radii:
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0 

    lsh = LSH(train_features)
    lsh.train(num_vector=10, seed=51)
    
    for query_index in range(len(train_features)):
        
        query_vec = train_features[query_index]
        all_neighbors = lsh.query(query_vec, radius)
        all_neighbors = all_neighbors.iloc[1:]

        true_label = train_labels[query_index] # true label of image
        neighbor_labels = [train_labels[i] for i in all_neighbors['id']]  # add all neighbor labels to a list

        # Apply same logic as to kNN to determine metrics

        if len(neighbor_labels) > 0:
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
    #if precision >= final_precision:
        best_recall = recall
        best_radius = radius
        final_precision = precision
        final_f1_score = f1
        
print(f"Best radius: {best_radius} with best recall: {best_recall}, precision:{final_precision} and f1_score:{final_f1_score}")

# np.save('Brecalls/recalls51.npy', recalls)
# np.save('Bprecisions/precisions51.npy', precisions)
# np.save('Bf1scores/f1scores51.npy', f1_scores)

          
    
    

