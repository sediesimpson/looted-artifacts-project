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
def split_dataset(dataset, test_split=0.2, val_split=0, shuffle=True, random_seed=42):
    # Identify unique images by their basename
    unique_images = list(set(os.path.basename(path) for path in dataset.img_paths))
    
    # Split the dataset based on unique images
    train_val_imgs, test_imgs = train_test_split(unique_images, test_size=test_split, random_state=random_seed)
    train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_split/(1-test_split), random_state=random_seed)
    
    train_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in train_imgs]
    val_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in val_imgs]
    test_indices = [i for i, path in enumerate(dataset.img_paths) if os.path.basename(path) in test_imgs]
    
    return train_indices, val_indices, test_indices
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
batch_size = 32
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
test_split = 0.1

# Create dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD10A"
dataset = CustomImageDataset2(root_dir)

# Get label information
label_info = dataset.get_label_info()
print("Label Information:", label_info)

# Get the number of images per label
label_counts = dataset.count_images_per_label()
print("Number of images per label:", label_counts)

# Split dataset without data leakage
train_indices, val_indices, test_indices = split_dataset(dataset, test_split=0.2, val_split=0.1, shuffle=True, random_seed=42)
    
# Create data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
# # Create dataset
# root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q"
# dataset = CustomImageDataset2(root_dir)

# # Get label information
# label_info = dataset.get_label_info()
# print("Label Information:", label_info)

# # Get the number of images per label
# label_counts = dataset.count_images_per_label()
# print("Number of images per label:", label_counts)

# train_loader = DataLoader(dataset, batch_size=1)

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
#num_classes = 28
hidden_features = 512
model = CustomResNet50(hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training set 
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)

# Extract features for test set 
test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

#----------------------------------------------------------------------------------------------------------------------------
# Hashing Function for LSH - Random Hashing Used
#----------------------------------------------------------------------------------------------------------------------------
def random_projection_hash(vector, num_hashes=10, dim=2048):  # Ensure `dim` matches feature vector size
    vector = np.array(vector)  # Ensure vector is a numpy array
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)  # Reshape to (1, dim) if vector is 1D
    projections = np.random.randn(num_hashes, dim)
    hash_codes = (np.dot(vector, projections.T) > 0).astype(int).flatten()
    return hash_codes

# def random_projection_hash(vector, num_hashes=10, dim=2048):
#     projections = np.random.randn(num_hashes, dim)
#     hash_codes = (np.dot(vector, projections.T) > 0).astype(int)
#     return hash_codes.flatten()

#----------------------------------------------------------------------------------------------------------------------------
# Building the hash tables 
#----------------------------------------------------------------------------------------------------------------------------
def build_hash_tables(hashes):
    hash_tables = defaultdict(list)
    for idx, hash_code in enumerate(hashes):
        hash_code_str = ''.join(map(str, hash_code))
        hash_tables[hash_code_str].append(idx)
    return hash_tables

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

# def query_image(image_path, model, hash_tables, preprocessed_images, device, k=5):
#     query_img = preprocess_image(image_path).unsqueeze(0).to(device)  # Add batch dimension and move to device
#     query_features = extract_features_single(query_img, model, device)  # Ensure this returns (1, 2048) array
#     query_hash = random_projection_hash(query_features, num_hashes=10, dim=2048)
    
#     # Ensure query_hash is 2D
#     if query_hash.ndim == 1:
#         query_hash = query_hash.reshape(1, -1)
    
#     # Flatten and convert each row of hash_codes to a string for hash table lookup
#     query_hash_strs = [''.join(map(str, row)) for row in query_hash]
    
#     candidates = []
#     for query_hash_str in query_hash_strs:
#         candidates.extend(hash_tables.get(query_hash_str, []))
    
#     if not candidates:
#         return []
    
#     # Return top-k similar images based on some distance measure (e.g., Hamming distance)
#     # Return unique candidates for simplicity
#     return [preprocessed_images[i] for i in set(candidates)]

def query_images(query_image_paths, model, hash_tables, preprocessed_images, device, k=5):
    all_query_hashes = []
    root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD10A"

    for image_path in query_image_paths:
        image_path = os.path.join(root_dir, image_path)
        print(image_path)
        query_img = preprocess_image(image_path).unsqueeze(0).to(device)  # Add batch dimension and move to device
        query_features = extract_features_single(query_img, model, device)
        query_hash = random_projection_hash(query_features, num_hashes=10, dim=2048)
        
        if query_hash.ndim == 1:
            query_hash = query_hash.reshape(1, -1)
        
        query_hash_strs = [''.join(map(str, row)) for row in query_hash]
        all_query_hashes.extend(query_hash_strs)
    
    candidates = []
    for query_hash_str in all_query_hashes:
        if query_hash_str in hash_tables:
            print(f"Found candidates for hash {query_hash_str}: {hash_tables[query_hash_str]}")
        candidates.extend(hash_tables.get(query_hash_str, []))
    
    if not candidates:
        print("No candidates found.")
        return []
    
    # Count the frequency of each candidate
    candidate_counts = {}
    for candidate in candidates:
        if candidate in candidate_counts:
            candidate_counts[candidate] += 1
        else:
            candidate_counts[candidate] = 1
    
    # Sort candidates by frequency and return the top-k
    sorted_candidates = sorted(candidate_counts.items(), key=lambda item: item[1], reverse=True)
    closest_candidates = [preprocessed_images[idx] for idx, _ in sorted_candidates[:k]]
    
    print("Closest Candidates:", closest_candidates)
    return closest_candidates

#----------------------------------------------------------------------------------------------------------------------------
# Example Implementation
#----------------------------------------------------------------------------------------------------------------------------
# folder_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q"

# query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/la_data/Accessories/Berge, Paris, 10-10-2017, Lot 221.jpg"
# query_image_path  = preprocess_image(query_image_path).unsqueeze(0).to(device)
# query_features = extract_features_single(query_image_path, model, device)
# query_hash = random_projection_hash(query_features, num_hashes=10, dim=2048)
# print(query_features)
# print(query_hash)

# # Ensure query_hash is 2D
# if query_hash.ndim == 1:
#     query_hash = query_hash.reshape(1, -1)

# print(query_hash)

# Flatten and convert each row of hash_codes to a string for hash table lookup
# query_hash_strs = [''.join(map(str, row)) for row in query_hash]
# print(query_hash_strs) 

hashes = [random_projection_hash(f, num_hashes=10, dim=2048) for f in train_features]  
hash_tables = build_hash_tables(hashes)
print(hash_tables)

    
# candidates = []
# for query_hash_str in query_hash_strs:
#     candidates.extend(hash_tables.get(query_hash_str, []))
#     print(candidates)
# sys.exit()






# hashes = [random_projection_hash(f, num_hashes=10, dim=2048) for f in train_features]  
# hash_tables = build_hash_tables(hashes)

# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return transform(image)


# # Query an image
# # query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Accessories - Query /Berge, Paris, 1-12-2011, Lot 272.jpg"
# # query_img = preprocess_image(query_image_path).unsqueeze(0).to(device)  # Add batch dimension and move to device
# # similar_images = query_image(query_img, model, hash_tables, train_img_paths)
# # print(similar_images)

# # Query an image
# #query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Accessories - Query /Berge, Paris, 1-12-2011, Lot 272.jpg"
# query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/la_data/Accessories/Berge, Paris, 10-10-2017, Lot 221.jpg"

# Query multiple images with debugging information
similar_images = query_images(test_img_paths, model, hash_tables, train_img_paths, device)

if similar_images:
    print("Similar Images:", similar_images)
else:
    print("No similar images found.")