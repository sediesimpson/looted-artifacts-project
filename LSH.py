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
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
# Create dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q"
dataset = CustomImageDataset2(root_dir)

# Get label information
label_info = dataset.get_label_info()
print("Label Information:", label_info)

# Get the number of images per label
label_counts = dataset.count_images_per_label()
print("Number of images per label:", label_counts)

train_loader = DataLoader(dataset, batch_size=1)

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
# def hamming_distance(hash1, hash2):
#     return np.sum(hash1 != hash2)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def query_image(image_path, model, hash_tables, preprocessed_images, device, k=5):
    query_img = preprocess_image(image_path).unsqueeze(0).to(device)  # Add batch dimension and move to device
    query_features = extract_features_single(query_img, model, device)  # Ensure this returns (1, 2048) array
    query_hash = random_projection_hash(query_features, num_hashes=10, dim=2048)
    
    # Ensure query_hash is 2D
    if query_hash.ndim == 1:
        query_hash = query_hash.reshape(1, -1)
    
    # Flatten and convert each row of hash_codes to a string for hash table lookup
    query_hash_strs = [''.join(map(str, row)) for row in query_hash]
    
    candidates = []
    for query_hash_str in query_hash_strs:
        candidates.extend(hash_tables.get(query_hash_str, []))
    
    if not candidates:
        return []
    
    # Return top-k similar images based on some distance measure (e.g., Hamming distance)
    # Return unique candidates for simplicity
    return [preprocessed_images[i] for i in set(candidates)]

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
    
#     # Calculate Hamming distance between the query hash and each candidate's hash
#     distances = []
#     for candidate_idx in set(candidates):
#         candidate_hash = random_projection_hash(train_features[candidate_idx], num_hashes=10, dim=2048)
#         dist = hamming_distance(query_hash, candidate_hash)
#         distances.append((candidate_idx, dist))
    
#     # Sort candidates by Hamming distance and return the top-k
#     distances.sort(key=lambda x: x[1])
#     closest_candidates = [preprocessed_images[idx] for idx, _ in distances[:k]]
    
#     return closest_candidates



#----------------------------------------------------------------------------------------------------------------------------
# Example Implementation
#----------------------------------------------------------------------------------------------------------------------------
folder_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q"

query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/la_data/Accessories/Berge, Paris, 10-10-2017, Lot 221.jpg"
query_image_path  = preprocess_image(query_image_path).unsqueeze(0).to(device)
query_features = extract_features_single(query_image_path, model, device)
query_hash = random_projection_hash(query_features, num_hashes=10, dim=2048)
print(query_features)
print(query_hash)

# Ensure query_hash is 2D
if query_hash.ndim == 1:
    query_hash = query_hash.reshape(1, -1)

print(query_hash)

# Flatten and convert each row of hash_codes to a string for hash table lookup
query_hash_strs = [''.join(map(str, row)) for row in query_hash]
print(query_hash_strs) 

hashes = [random_projection_hash(f, num_hashes=10, dim=2048) for f in train_features]  
hash_tables = build_hash_tables(hashes)
print(hash_tables)

    
candidates = []
for query_hash_str in query_hash_strs:
    candidates.extend(hash_tables.get(query_hash_str, []))
    print(candidates)
sys.exit()






hashes = [random_projection_hash(f, num_hashes=10, dim=2048) for f in train_features]  
hash_tables = build_hash_tables(hashes)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


# Query an image
# query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Accessories - Query /Berge, Paris, 1-12-2011, Lot 272.jpg"
# query_img = preprocess_image(query_image_path).unsqueeze(0).to(device)  # Add batch dimension and move to device
# similar_images = query_image(query_img, model, hash_tables, train_img_paths)
# print(similar_images)

# Query an image
#query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Accessories - Query /Berge, Paris, 1-12-2011, Lot 272.jpg"
query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/la_data/Accessories/Berge, Paris, 10-10-2017, Lot 221.jpg"
similar_images = query_image(query_image_path, model, hash_tables, train_img_paths, device)
if similar_images:
    print("Similar Images:", similar_images)
else:
    print("No similar images found.")