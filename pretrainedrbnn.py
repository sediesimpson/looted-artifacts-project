import os
import sys
from tqdm import tqdm
import numpy as np

from dataloaderdup import *


from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torchvision import models, transforms


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from sklearn.metrics import precision_score, recall_score, f1_score, pairwise_distances
from sklearn.utils import shuffle

import math
import seaborn as sns

from pprint import pprint
#------------------------------------------------------------------------------------------------------------------------
# Define CustomResNetClassifier
#------------------------------------------------------------------------------------------------------------------------
class ResNet50Extractor(nn.Module):
    def __init__(self, weights=None):
        super(ResNet50Extractor, self).__init__()
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
#train_dataset = CustomImageDatasetTrain(root_dir)
test_dataset = CustomImageDatasetTest(root_dir)

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
# Model
#------------------------------------------------------------------------------------------------------------------------
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = ResNet50Extractor(weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_label_indices, train_img_paths, train_labels, train_label_counts = extract_features(train_loader, model, device)
test_features, test_label_indices, test_img_paths, test_labels, test_label_counts = extract_features(test_loader, model, device)

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

# plt.hist(distribution, color = 'mediumseagreen', density=True)
# plt.xlabel('Radius')
# plt.ylabel('Density')

# Overlay a KDE line

fig, ax = plt.subplots()
dist_plot = sns.kdeplot(distribution, color='teal')

ax.set_xlabel('Radius')
ax.set_ylabel('Density')

lines_for_plot = dist_plot.get_lines()
print('lines for plot:',lines_for_plot)

for line in lines_for_plot:
    # ax.plot(line.get_data())
    x, y = line.get_data()
    print(x[np.argmax(y)])
    ax.axvline(x[np.argmax(y)], ls='--', color='black')

plt.savefig('Aplotsknn/distribution_fixed.png')

#----------------------------------------------------------------------------------------------------------------------------
# Calculate average recall and precision per class
#----------------------------------------------------------------------------------------------------------------------------
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0 
precision = 0
recall = 0

per_class_results = {}

nbrs = NearestNeighbors(metric='cosine', algorithm='brute', radius=0.35).fit(test_features)

for query_index in range(len(test_features)):
    distances, indices = nbrs.radius_neighbors([test_features[query_index]], sort_results=True)

    distances = distances[0] # remove first distance as it is itself
    indices = indices[0] # remove first index as it is itself

    true_label = test_labels[query_index]
    true_label_idx = test_label_indices[query_index]
    true_label_counts = test_label_counts[query_index].item()

    neighbor_labels = test_label_indices[indices]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if true_label == 'Αγάλματα Γυναικεία Γυμνά 3':
        print(f'number of neighbours={len(neighbor_labels)}')
    for label in neighbor_labels:
       # print(f"neighbour label: {label}, true label: {true_label_idx}")
        if label == true_label_idx:
            tp += 1
        else:
            fp += 1

    fn = true_label_counts - tp
    tn = len(test_features) - len(neighbor_labels) - fn

    # print(f"label: {test_labels[query_index]}, label count: {test_label_counts[query_index]}, tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
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

print("Overall results")
results = {
    "true positives": true_positives,
    "false positives": false_positives,
    "true negatives": true_negatives,
    "false negatives": false_negatives,
    "recall": recall,
    "precision": precision
}
pprint(results)

precisions = []
recalls = []

print()
print("Per class results")
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
    
    # if precision == 1.0:
    #     print(f'precision=1, label={label}, number of true positives={true_positives}')

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
    print(f"===== LABEL: {label} =====")
    pprint(results)
    print()

print(np.mean(recalls))
print(np.mean(precisions))

