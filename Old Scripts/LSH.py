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
#------------------------------------------------------------------------------------------------------------------------
# Define the CustomClassifier module
#------------------------------------------------------------------------------------------------------------------------
class CustomClassifier(nn.Module):
    def __init__(self, num_features, hidden_features):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CustomResNet50(nn.Module):
    def __init__(self, hidden_features):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze the convolutional layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        num_features = self.resnet50.fc.in_features  # This should be 2048 for ResNet50
        self.resnet50.fc = nn.Identity()  # Remove the existing fully connected layer
        self.custom_classifier = CustomClassifier(num_features, hidden_features)

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.resnet50.avgpool(self.resnet50.layer4(self.resnet50.layer3(self.resnet50.layer2(self.resnet50.layer1(self.resnet50.maxpool(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x)))))))))
        x = torch.flatten(x, 1)
        x = self.custom_classifier(x)
        return x

#--------------------------------------------------------------------------------------------------------------------------
# Split dataset 
#--------------------------------------------------------------------------------------------------------------------------
# Function to ensure no data leakage
def split_dataset(dataset, test_split=0.2, shuffle=True, random_seed=42):
    # Identify unique images by their basename
    unique_images = list(set(os.path.basename(path) for path in dataset.img_paths))
    
    # Split the dataset based on unique images
    train_imgs, test_imgs = train_test_split(unique_images, test_size=test_split, random_state=random_seed)
    
    train_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in train_imgs]
    test_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in test_imgs]
    
    return train_indices, test_indices

#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
batch_size = 1
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
test_split = 0.1

# Create dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD4A"
dataset = CustomImageDataset2(root_dir)

# Get label information
label_info = dataset.get_label_info()
print("Label Information:", label_info)

# Get the number of images per label
label_counts = dataset.count_images_per_label()
print("Number of images per label:", label_counts)

# Split dataset without data leakage
train_indices, test_indices = split_dataset(dataset, test_split=0.2, shuffle=True, random_seed=42)
    
# Create data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
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


def extract_features_single(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        features = model.resnet50.avgpool(
            model.resnet50.layer4(
                model.resnet50.layer3(
                    model.resnet50.layer2(
                        model.resnet50.layer1(
                            model.resnet50.maxpool(
                                model.resnet50.relu(
                                    model.resnet50.bn1(
                                        model.resnet50.conv1(image)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        features = torch.flatten(features, 1)
    return features.cpu().numpy()

#----------------------------------------------------------------------------------------------------------------------------
# Feature Extraction 
#----------------------------------------------------------------------------------------------------------------------------
hidden_features = 512
model = CustomResNet50(hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training set 
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)

# Extract features for test set 
test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

#----------------------------------------------------------------------------------------------------------------------------
# Hashing the query image 
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

    def query(self, query_vec, k, max_search_radius, initial_candidates=set()):
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
        # candidates = data[np.array(list(candidate_set)), :]
        # Reshape query_vec to be a 2D array
        query_vec = query_vec.reshape(1, -1)
        
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec, metric='cosine').flatten()

        return nearest_neighbors.nsmallest(k, 'distance')
    
# Function to show and save images
def show_images_and_save(query_image_path, neighbor_image_paths, save_path):
    fig, axes = plt.subplots(1, len(neighbor_image_paths) + 1, figsize=(20, 5))
    if len(neighbor_image_paths) + 1 == 1:
        axes = [axes] 
    
    # fig, axes = plt.subplots(1, len(neighbor_image_paths) + 1, figsize=(20, 5))
    fig.suptitle('Query Image and Nearest Neighbors', fontsize=16)

    # Display query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # Display nearest neighbor images
    for i, neighbor_image_path in enumerate(neighbor_image_paths):
        neighbor_img = Image.open(neighbor_image_path)
        axes[i + 1].imshow(neighbor_img)
        axes[i + 1].set_title(f'Neighbor {i+1}')
        axes[i + 1].axis('off')

    # Save the plot
    plt.savefig(save_path)
    plt.close(fig)
    
#----------------------------------------------------------------------------------------------------------------------------
# Implementation
#----------------------------------------------------------------------------------------------------------------------------
# Initialize LSH with extracted features
lsh = LSH(train_features)
lsh.train(num_vector=10, seed=42)  # Adjust num_vector based on your requirements

# Define the root directory where images are stored
save_dir = "lshplotscorrected/"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Define the top_k number of queries to process
top_k_queries = 2  
selected_query_indices = range(top_k_queries)

# Iterate through each data point in the test set
for query_index in range(len(test_features)):

    query_vec = test_features[query_index]  # Use the query vector from the data

    # Measure the query time
    start_time = time.time()

    # Find the k-nearest neighbors
    # k = 5
    # max_search_radius = 2
    # nearest_neighbors = lsh.query(query_vec, k, max_search_radius)
    
    # # Ensure the indices are integers
    # neighbor_indices = nearest_neighbors['id'].astype(int).tolist()

    # Find the k-nearest neighbors
    max_search_radius = 2
    all_neighbors = lsh.query(query_vec, len(train_features), max_search_radius)

    # Select the top 5 most confident predictions (smallest distances)
    top_k = 2
    top_neighbors = all_neighbors.nsmallest(top_k, 'distance')
    print(top_neighbors)

    # Ensure the indices are integers
    neighbor_indices = top_neighbors['id'].astype(int).tolist()
    distances = top_neighbors['distance'].tolist()

    # End time for the query
    end_time = time.time()
    query_time = end_time - start_time

    # Get the image paths of the query and its nearest neighbors
    query_image_path = os.path.join(root_dir, test_img_paths[query_index])
    neighbor_image_paths = [os.path.join(root_dir, train_img_paths[idx]) for idx in neighbor_indices]

    # Print the paths
    print(f"Query Image: {query_image_path}")
    print("Nearest Neighbors:")
    for path in neighbor_image_paths:
        print(path)
    print("\n")

    #     # Print the paths and distances
    # print(f"Query Image: {query_image_path}")
    # print("Nearest Neighbors and Distances:")
    # for path, distance in zip(neighbor_image_paths, distances):
    #     print(f"{path} - Distance: {distance}")
    # print("\n")


    # Define the save path for the plot
    #save_path = os.path.join(save_dir, f'query_{query_index}_neighbors.png')


    # Visualize and save the query image and its nearest neighbors
    #show_images_and_save(query_image_path, neighbor_image_paths, save_path)

