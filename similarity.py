import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from finaldataloader import CustomImageDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import cv2
#------------------------------------------------------------------------------------------------------------------------
# Define the CustomClassifier module
#------------------------------------------------------------------------------------------------------------------------
class CustomClassifier(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#------------------------------------------------------------------------------------------------------------------------
# Define the CustomResnet50 module
#------------------------------------------------------------------------------------------------------------------------
class CustomResNet50(nn.Module):
    def __init__(self, num_classes, hidden_features):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze the convolutional layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        num_features = self.resnet50.fc.in_features  # This should be 2048 for ResNet50
        self.custom_classifier = CustomClassifier(num_features, hidden_features, num_classes)

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.resnet50.avgpool(self.resnet50.layer4(self.resnet50.layer3(self.resnet50.layer2(self.resnet50.layer1(self.resnet50.maxpool(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x)))))))))
        x = torch.flatten(x, 1)
        x = self.custom_classifier(x)
        return x

#------------------------------------------------------------------------------------------------------------------------
# Define the feature extraction function
#------------------------------------------------------------------------------------------------------------------------

def extract_features(dataloader, model, device):
    model.eval()
    features_list = []
    labels_list = []
    img_paths_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting features"):
            img_tensors = []
            for img, label in zip(imgs, labels):
                img_tensors.append(img.to(device))  # Move images to device

            if not img_tensors:
                continue

            imgs = torch.stack(img_tensors)  # Convert list of tensors to a single tensor
            features = model.resnet50.avgpool(model.resnet50.layer4(model.resnet50.layer3(model.resnet50.layer2(model.resnet50.layer1(model.resnet50.maxpool(model.resnet50.relu(model.resnet50.bn1(model.resnet50.conv1(imgs)))))))))
            features = torch.flatten(features, 1)  # Flatten the features
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            img_paths_list.extend([path for path, _ in dataloader.dataset.img_paths])
    return np.vstack(features_list), np.hstack(labels_list), img_paths_list

#------------------------------------------------------------------------------------------------------------------------
# Define the query image extraction function
#------------------------------------------------------------------------------------------------------------------------
def extract_query_features(image_path, dataset, model, device):
    model.eval()
    with torch.no_grad():
        foreground_image = dataset.background_subtraction(image_path)  # Apply background subtraction
        if foreground_image is None:
            return None, None
        img_t = dataset.transform(foreground_image).unsqueeze(0).to(device)  # Preprocess, add batch dimension, and move to device
        features = model.resnet50.avgpool(model.resnet50.layer4(model.resnet50.layer3(model.resnet50.layer2(model.resnet50.layer1(model.resnet50.maxpool(model.resnet50.relu(model.resnet50.bn1(model.resnet50.conv1(img_t)))))))))
        features = torch.flatten(features, 1)  # Flatten the features

    # Find the label of the query image
    query_label = None
    for path, label in dataset.img_paths:
        if path == image_path:
            query_label = label
            break

    return features.cpu().numpy().squeeze(), query_label  # Remove batch dimension and convert to numpy array

#------------------------------------------------------------------------------------------------------------------------
# Define the similarity function
#------------------------------------------------------------------------------------------------------------------------

def compute_similarities(query_features, dataset_features):
    similarities = [cosine(query_features, features) for features in dataset_features]
    return similarities

#------------------------------------------------------------------------------------------------------------------------
# Define top N similar function
#------------------------------------------------------------------------------------------------------------------------
def retrieve_top_n_similar(similarities, img_paths, query_image_path, n=5):
    sorted_indices = np.argsort(similarities)
    top_n_indices = []
    for idx in sorted_indices:
        if img_paths[idx] != query_image_path:
            top_n_indices.append(idx)
        if len(top_n_indices) == n:
            break
    return top_n_indices


#------------------------------------------------------------------------------------------------------------------------
# Define the preprocess function
#------------------------------------------------------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#------------------------------------------------------------------------------------------------------------------------
# Load dataset
#------------------------------------------------------------------------------------------------------------------------

dataset_dir = "/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 5"
dataset = CustomImageDataset(root_dir=dataset_dir)

#------------------------------------------------------------------------------------------------------------------------
# Create dataloader
#------------------------------------------------------------------------------------------------------------------------
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 3
hidden_features = 2
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for whole dataset
features, labels, img_paths = extract_features(dataloader, model, device)

# Print shape of extracted features
print("Shape of extracted features:", features.shape)
print("Shape of labels:", labels.shape)

#------------------------------------------------------------------------------------------------------------------------
# Query, similarity and top N images
#------------------------------------------------------------------------------------------------------------------------
query_image_path = "/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 5/Accessories/Barakat Volume-11, FZ210.JPG"
query_features, query_label = extract_query_features(query_image_path, dataset, model, device)
similarities = compute_similarities(query_features, features)
n = 4
top_n_indices = retrieve_top_n_similar(similarities, img_paths, query_image_path, n=n)
print("Top N similar image indices:", top_n_indices)
print("Top N similar image indices:", top_n_indices)

#------------------------------------------------------------------------------------------------------------------------
# Visualise similar images
#------------------------------------------------------------------------------------------------------------------------

# Retrieve file paths and labels of the top N similar images
top_n_image_paths = [img_paths[i] for i in top_n_indices]
top_n_labels = [labels[i] for i in top_n_indices]
print(top_n_image_paths)


def visualize_similar_images(query_image_path, top_n_image_paths, top_n_labels, n=n):
    plt.figure(figsize=(15, 5))
    # Plot the query image
    query_img = Image.open(query_image_path).convert('RGB')
    plt.subplot(1, n+1, 1)
    plt.imshow(query_img)
    plt.title(f"Query Image (Label: {query_label})")
    plt.axis('off')
    # Plot the top N similar images
    for i, (img_path, label) in enumerate(zip(top_n_image_paths, top_n_labels), 1):
        img = Image.open(img_path).convert('RGB')
        plt.subplot(1, n+1, i+1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.show()
visualize_similar_images(query_image_path, top_n_image_paths, top_n_labels, n=n)
