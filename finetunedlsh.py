import os
import sys
from tqdm import tqdm
import numpy as np

from dataloaderdup import *


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

import torch
import torch.nn as nn
from torchvision import models, transforms


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from copy import copy

from sklearn.metrics import precision_score, recall_score, f1_score, pairwise_distances
from sklearn.utils import shuffle

import math
import seaborn as sns

from pprint import pprint
import torch.optim as optim


from collections import defaultdict
from pandas import DataFrame
from itertools import combinations
import pprint
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
        print(f'number of buckets: {len(table)}, length of data: {len(self.data)}')
        return self

    def __search_nearby_bins(self, query_bin_bits, table, search_radius, initial_candidates=set()):
        #print('in search bins function, the search radius:', search_radius)
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        candidate_set = copy(initial_candidates)

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
        
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec, metric='cosine').flatten()

        # Sort the DataFrame by the 'distance' column
        nearest_neighbors = nearest_neighbors.sort_values(by='distance', ascending=True)

        return nearest_neighbors

    def hamming_weight(self, x):
        x -= (x >> 1) & 0x5555555555555555
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
        return ((x * 0x0101010101010101) & 0xffffffffffffffff ) >> 56

    def query2(self, query_vec, max_search_radius):
        if not self.model:
            raise ValueError('Model not yet built. Exiting!')
        
        data = self.data
        table = self.model['table']
        random_vectors = self.model['random_vectors']
        num_vector = self.model['num_vector']

        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
        qv = query_vec
        query_vec = (query_vec.dot(random_vectors) >= 0).flatten()
        query = query_vec.dot(powers_of_two).item()


        candidate_set = set()
    

        for (bucket, candidates) in table.items():
            hamming = query ^ bucket
            hamming = hamming.item()
            # print('hamming:',type(hamming))
            # print('bucket:', bucket)
            hw = self.hamming_weight(hamming)
            
            if hw <= max_search_radius:
                candidate_set.update(candidates)

        # Ensure the indices are integers before using them
        candidate_set = set(map(int, candidate_set))

        # Handle the case where no candidates are found
        if not candidate_set:
            return DataFrame({'id': [], 'distance': []})
        
        #print(candidate_set.shape)

        nearest_neighbors = DataFrame({'id': list(candidate_set)})

        # Reshape query_vec to be a 2D array
        qv = qv.reshape(1, -1)

        candidates = data[np.array(list(candidate_set), dtype=int), :]
        nearest_neighbors['distance'] = pairwise_distances(candidates, qv, metric='cosine').flatten()
        nearest_neighbors = nearest_neighbors.sort_values(by='distance', ascending=True)

        return nearest_neighbors

#----------------------------------------------------------------------------------------------------------------------------
# Optimal radius determination 
#----------------------------------------------------------------------------------------------------------------------------
max_search_radius = 16
lsh = LSH(train_features)
lsh.train(num_vector=16, seed=42) 

for search_radius in range(max_search_radius):

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0 
    precision = 0
    recall = 0

    per_class_results = {}
    precisions = []
    recalls = []

    for query_index in range(len(train_features)):
        query_vec = train_features[query_index]
        all_neighbors = lsh.query2(query_vec, search_radius)

        indices = all_neighbors['id'].astype(int).tolist()
        true_label = train_labels[query_index]
        neighbor_labels = train_labels[indices]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for label in neighbor_labels:
            if label == true_label:
                tp += 1
            else:
                fp += 1

        fn = train_label_counts[query_index] - tp
        tn = len(train_features) - len(neighbor_labels) - fn

        if true_label in per_class_results:
            results = per_class_results[true_label]
            results["true positives"] += tp
            results["false positives"] += fp
            results["true negatives"] += tn
            results["false negatives"] += fn
        else:
            results = {
                "true positives": tp,
                "false positives": fp,
                "true negatives": tn,
                "false negatives": fn,
            }
            per_class_results[true_label] = results

        # accumulate
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        true_negatives += tn


        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives) 
        else:
            precision = 0

        results = {
            "true positives": true_positives,
            "false positives": false_positives,
            "true negatives": true_negatives,
            "false negatives": false_negatives,
            "recall": recall,
            "precision": precision
        }

        pprint(results)

        print()
        print("Per class results")
        print(per_class_results)
    for label, results in per_class_results.items():
        true_positives = results["true positives"]
        false_positives = results["false positives"]
        true_negatives = results["true negatives"]
        false_negatives = results["false negatives"]

        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives) 
        else:
            precision = 0

        results = {
            "true positives": true_positives,
            "false positives": false_positives,
            "true negatives": true_negatives,
            "false negatives": false_negatives,
            "recall": recall,
            "precision": precision
        }
        precisions.append(precision)
        recalls.append(recall)

#----------------------------------------------------------------------------------------------------------------------------
# Testing the precision and recall for LSH 
#----------------------------------------------------------------------------------------------------------------------------
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0 
precision = 0
recall = 0
per_class_results = {}

lsh = LSH(test_features)
lsh.train(num_vector=16, seed=42) 

for query_index in range(len(test_features)):
    max_search_radius = 5
    query_vec = test_features[query_index]
    all_neighbors = lsh.query2(query_vec, max_search_radius)

    indices = all_neighbors['id'].astype(int).tolist()
    true_label = test_labels[query_index]
    neighbor_labels = test_labels[indices]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label in neighbor_labels:
        if label == true_label:
            tp += 1
        else:
            fp += 1

    fn = test_label_counts[query_index] - tp
    tn = len(test_features) - len(neighbor_labels) - fn

    if true_label in per_class_results:
        results = per_class_results[true_label]
        results["true positives"] += tp
        results["false positives"] += fp
        results["true negatives"] += tn
        results["false negatives"] += fn
    else:
        results = {
            "true positives": tp,
            "false positives": fp,
            "true negatives": tn,
            "false negatives": fn,
        }
        per_class_results[true_label] = results

    # accumulate
    true_positives += tp
    false_positives += fp
    false_negatives += fn
    true_negatives += tn


if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0

if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives) 
else:
    precision = 0

results = {
    "true positives": true_positives,
    "false positives": false_positives,
    "true negatives": true_negatives,
    "false negatives": false_negatives,
    "recall": recall,
    "precision": precision
}

#pprint(results)

precisions = []
recalls = []

print()
print("Per class results")
print(per_class_results)
for label, results in per_class_results.items():
    true_positives = results["true positives"]
    false_positives = results["false positives"]
    true_negatives = results["true negatives"]
    false_negatives = results["false negatives"]

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives) 
    else:
        precision = 0

    results = {
        "true positives": true_positives,
        "false positives": false_positives,
        "true negatives": true_negatives,
        "false negatives": false_negatives,
        "recall": recall,
        "precision": precision
    }
    precisions.append(precision)
    recalls.append(recall.item())
print(precisions)
print(recalls)
