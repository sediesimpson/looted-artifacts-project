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

        # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                table[bin_index] = []
            table[bin_index].append(data_index)

        self.model = {'bin_indices': bin_indices, 'table': table, 'random_vectors': random_vectors, 'num_vector': num_vector}
        return self

    def __search_nearby_bins(self, query_bin_bits, table, search_radius, initial_candidates=set()):
        #print('in search bins function, the search radius:', search_radius)
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
        buckets = self.model["bin_indices"]

        query = query_bin_bits.dot(powers_of_two)
        for bucket in buckets:
            hamming = query - bucket
            # if the hamming is zero or a power of two, then we have a 
            if (hamming & (hamming-1) == 0):



        # candidate_set = copy(initial_candidates)
        candidate_set = set()


        for different_bits in combinations(range(num_vector), search_radius):
            #print('Different bits:', different_bits)
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
            cs = self.__search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=initial_candidates)
            candidate_set.update(cs)

        # Ensure the indices are integers before using them
        candidate_set = set(map(int, candidate_set))

        # Handle the case where no candidates are found
        if not candidate_set:
            return DataFrame({'id': [], 'distance': []})

        candidates = data[np.array(list(candidate_set), dtype=int), :]

        nearest_neighbors = DataFrame({'id': list(candidate_set)})

        # Reshape query_vec to be a 2D array
        query_vec = query_vec.reshape(1, -1)
        
        # We actually don't care here which are the nearest, this should be optional
        # All we are care about is getting the distances we believe are the same
        # nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec, metric='cosine').flatten()

        # Sort the DataFrame by the 'distance' column
        # nearest_neighbors = nearest_neighbors.sort_values(by='distance', ascending=True)

        return nearest_neighbors


def show_images_and_save(query_image_path, neighbor_image_paths, save_path):
    # Define the maximum number of images per row
    max_per_row = 5
    total_images = len(neighbor_image_paths) + 1
    rows = (total_images // max_per_row) + int(total_images % max_per_row != 0)
    cols = min(total_images, max_per_row)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # Flatten the axes array if it is multidimensional, to handle it uniformly
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    
    # Display query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # Display nearest neighbor images
    for i, neighbor_image_path in enumerate(neighbor_image_paths):
        img = Image.open(neighbor_image_path)
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f'Neighbour {i+1}')
        axes[i + 1].axis('off')

    # Hide any unused subplots
    for j in range(total_images, rows * cols):
        fig.delaxes(axes[j])

    # Save the plot
    plt.savefig(save_path)
    plt.close(fig)


# # Train LSH on the full training dataset
# lsh = LSH(train_features)
# lsh.train(num_vector=16, seed=42)
# query_index = 10
# max_search_radius = 1
# query_vec = train_features[10]
# all_neighbors = lsh.query(query_vec, max_search_radius)

# indices = all_neighbors['id'].astype(int).tolist()
# distances = all_neighbors['distance'].tolist()

# # Define the root directory where images are stored
# save_dir = "lshplotscorrected/"

# # Ensure the save directory exists
# os.makedirs(save_dir, exist_ok=True)

# query_image_path = os.path.join(root_dir, train_img_paths[query_index])
# neighbor_image_paths = [os.path.join(root_dir, train_img_paths[idx]) for idx in indices]
# print('==========query image path============')
# print(query_image_path)
# print('==========neighbor image path============')
# neighbor_image_paths = neighbor_image_paths[1:]
# print(neighbor_image_paths)

# save_path = 'lshplotscorrected/query10FT.pdf'
# show_images_and_save(query_image_path, neighbor_image_paths, save_path)


# sys.exit()

# #----------------------------------------------------------------------------------------------------------------------------
# # Try to optimise the search radius
# #----------------------------------------------------------------------------------------------------------------------------
# recalls = []
# precisions = []
# f1_scores = []
# # Iterate through each data point in the test set

# max_search_radius = 16

# TP = []
# FP = []
# TN = []
# FN = []

# for search_radius in range(max_search_radius):
#     lsh = LSH(train_features)
#     lsh.train(num_vector=16, seed=42)  # Informed by literature
#     print('search_radius:', search_radius)
#     correct = 0 
#     curve = []
#     true_positives = 0
#     false_positives = 0
#     true_negatives = 0
#     false_negatives = 0 

#     for query_index in range(len(train_features)):

#         num_correct = train_label_counts[query_index]

#         query_vec = train_features[query_index]  # Use the query vector from the data
#         # print('query vector:', query_vec)

#         all_neighbors = lsh.query(query_vec, search_radius)
#         #print(all_neighbors)

#         # Ensure the indices are integers
#         indices = all_neighbors['id'].astype(int).tolist()
#         distances = all_neighbors['distance'].tolist()

#         #print('indices:', indices)
#         # print('distances:', distances)

#         # Get the image paths of the query and its nearest neighbors
#         query_image_path = os.path.join(root_dir, train_img_paths[query_index])
#         neighbor_image_paths = [os.path.join(root_dir, train_img_paths[idx]) for idx in indices]

#        # if query_index == 0: 
#             # print('==========dataframe of all neighbours:==========')
#             # print(all_neighbors)

 
#             #print('==========number of candidates returned:==========')
#         #     print('number of candidates returned:',len(all_neighbors))
#         #     print('indices:', indices)
#         #     print('==========query image path:==========')
#         #     print(query_image_path)

#         #     print('==========neighbor image paths:==========')
#         #     print(neighbor_image_paths)
#         #     print()

#         # if query_index >=1:
#         #     break

#         true_label = train_labels[query_index]


#         if len(indices) > 0:
#             for label in train_labels[indices]:
#                 if label == true_label:
#                     true_positives +=1 
#                 else:
#                     true_negatives +=1 

#             fn = train_label_counts[query_index] - true_positives

#             if fn < 0:
#                 fn = 0

#             false_negatives += fn

#             fp = len(indices) - true_positives

#             if fp < 0:
#                 fp = 0

#             false_positives += fp

#     if (true_positives + false_negatives) > 0:
#         recall = true_positives / (true_positives + false_negatives)
#     else:
#         recall = 0

#     if (true_positives + false_positives) > 0:
#         precision = true_positives / (true_positives + false_positives) 
#     else:
#         precision = 0

#     if (precision + recall) > 0:
#         f1 = (2 * precision * recall) / (precision + recall)
#     else:
#         f1 = 0 

#     recalls.append(recall)
#     precisions.append(precision)
#     f1_scores.append(f1)

# print('recalls:', recalls)
# print('precisions:', precisions)
# print('f1 scores:', f1_scores)


# np.save('Bplotslsh/recallsLSH2.npy', recalls)
# np.save('Bplotslsh/precisionsLSH2.npy', precisions)
# np.save('Bplotslsh/f1scoresLSH2.npy', f1_scores)


#----------------------------------------------------------------------------------------------------------------------------
# Figuring out query time
#----------------------------------------------------------------------------------------------------------------------------
save_dir = 'Adatasetsizes/'
# Initialize lists to store dataset sizes and query times
dataset_sizes = []
query_times = []

subset_sizes = range(50,900,50)

# Number of queries to average for each subset size
num_queries = 10

lsh = LSH(train_features)
lsh.train(num_vector=16, seed=42)


# Iterate through each subset size
for subset_size in subset_sizes:
    # Subset the test dataset
    subset_indices = np.random.choice(len(train_features), subset_size, replace=False)
    subset_train_features = train_features[subset_indices]


      # Measure the query times for multiple queries
    total_query_time = 0
    for _ in range(num_queries):

        start_time = time.time()

        for query_index in range(len(subset_train_features)):
             # Find all the neighbors
            query_vec = subset_train_features[query_index]
            max_search_radius = 1
            all_neighbors = lsh.query(query_vec, max_search_radius)


        # End time for the query
        end_time = time.time()
        query_time = end_time - start_time

        # Accumulate the query time
        total_query_time += query_time

    # Calculate the average query time for this subset size
    average_query_time = total_query_time / num_queries
    dataset_sizes.append(subset_size)
    query_times.append(average_query_time)

    print(f"Subset Size: {subset_size}, Average Query Time: {average_query_time}")

# Optionally, you can save the dataset_sizes and query_times for further analysis
np.save(os.path.join(save_dir, "dataset_sizes_LSH_FT.npy"), np.array(dataset_sizes))
np.save(os.path.join(save_dir, "query_times_LSH_FT.npy"), np.array(query_times))