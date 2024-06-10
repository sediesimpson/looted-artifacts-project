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

from finaldataloader import *

from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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
#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 3
hidden_features = 512
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#------------------------------------------------------------------------------------------------------------
# New test image
#------------------------------------------------------------------------------------------------------------
def preprocess(image):
    # Define the transformations: resize, convert to tensor, and normalize
    preprocess_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size expected by your model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with the mean and std used during training
    ])
    
    # Apply the transformations
    image = preprocess_transforms(image)
    return image

# Assuming you have a function to preprocess and extract features from an image
# Function to extract features from a single image
def extract_features_from_image(image_path, model, device):
    # Load the image
    image = Image.open(image_path)
    
    # Preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Extract features using the model
    model.eval()
    with torch.no_grad():
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

# Load the k-NN classifier from the file
with open('knn_models/knn_classifier.pkl', 'rb') as f:
    knn_classifier = pickle.load(f)

with open('knn_models/train_img_paths.pkl', 'rb') as f:
    train_img_paths = pickle.load(f)

# Generate a timestamp to include in the log file name
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = "knn_query_logs"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")


# Load the new query image and extract features
new_query_image_path = '/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Inscriptions - Query/CHR PAR Mar.D. Serres Coll. 16.1 &17.2.11 148.5.JPG' 
new_query_features = extract_features_from_image(new_query_image_path, model, device)

# Predict the label for the new query image (if needed)
new_query_prediction = knn_classifier.predict(new_query_features)
predicted_label = new_query_prediction[0]
with open(log_file, 'a') as log:
    log.write(f"Predicted Label for the new query image: {predicted_label}\n")

# Get the nearest neighbors for the new query image
distances, indices = knn_classifier.kneighbors(new_query_features, n_neighbors=2)

# Print the paths of the nearest neighbors
with open(log_file, 'a') as log:
    log.write("Nearest neighbors for the new query image:\n")
    for neighbor_idx in indices[0]:
        neighbor_path = train_img_paths[neighbor_idx]
        log.write(f"Path: {neighbor_path}\n")



# List of paths to your 10 query images
query_image_paths = [
    'path/to/your/query/image1.jpg',
    'path/to/your/query/image2.jpg',
    'path/to/your/query/image3.jpg',
    # Add paths to all 10 images
]

# Load the new query images, extract features, and make predictions
with open(log_file, 'a') as log:
  for query_image_path in query_image_paths:
      new_query_features = extract_features_from_image(query_image_path, model, device)
      
      # Predict the label for the new query image
      new_query_prediction = knn_classifier.predict(new_query_features)
      predicted_label = new_query_prediction[0]
      log.write(f"Predicted Label for the query image {query_image_path}: {predicted_label}")
      
      # Get the nearest neighbors for the new query image
      distances, indices = knn_classifier.kneighbors(new_query_features, n_neighbors=5)
      
      # Print the paths of the nearest neighbors
      log.write(f"Nearest neighbors for the query image {query_image_path}:")
      for neighbor_idx in indices[0]:
          neighbor_path = train_img_paths[neighbor_idx]
          print(f"Path: {neighbor_path}")
      log.write()  # Add a newline for better readability