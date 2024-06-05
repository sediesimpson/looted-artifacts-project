import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from finaldataloader import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import sys
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from torchvision.models import ResNet50_Weights
import torch.optim as optim
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
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
    
#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
batch_size = 16
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
test_split = 0.1

# Create dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/Test Dataset 4"
dataset = CustomImageDataset(root_dir)

# Get label information
label_info = dataset.get_label_info()
print("Label Information:", label_info)

# Get the number of images per label
label_counts = dataset.count_images_per_label()
print("Number of images per label:", label_counts)

# Create data indices for training, validation, and test splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
test_split_idx = int(np.floor(test_split * dataset_size))
validation_split_idx = int(np.floor(validation_split * (dataset_size - test_split_idx)))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

test_indices = indices[:test_split_idx]
train_val_indices = indices[test_split_idx:]
train_indices = train_val_indices[validation_split_idx:]
val_indices = train_val_indices[:validation_split_idx]

# Create data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

def count_labels_in_loader(loader, class_to_idx):
    # Initialize label_counts using the class indices directly
    label_counts = {idx: 0 for idx in class_to_idx.values()}
    for _, labels, _ in loader:
        for label in labels.numpy():
            if label in label_counts:
                label_counts[label] += 1
    return label_counts

# Check label distribution in each loader
train_label_counts = count_labels_in_loader(train_loader, dataset.class_to_idx)
valid_label_counts = count_labels_in_loader(valid_loader, dataset.class_to_idx)
test_label_counts = count_labels_in_loader(test_loader, dataset.class_to_idx)

print("Training label distribution:", train_label_counts)
print("Validation label distribution:", valid_label_counts)
print("Test label distribution:", test_label_counts)
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
            img_paths_list.extend(img_paths)  # No need to iterate over each path individually

    # Convert lists to numpy arrays
    features_array = np.vstack(features_list)
    labels_array = np.hstack(labels_list)
    
    return features_array, labels_array, img_paths_list
#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 3
hidden_features = 512
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# #------------------------------------------------------------------------------------------------------------------------
# # Run KNN similarity 
# #------------------------------------------------------------------------------------------------------------------------
# Extract features for training and validation sets
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)
val_features, val_labels , val_img_paths = extract_features(valid_loader, model, device)

# Train the KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_classifier.fit(train_features, train_labels)

# Validate the classifier
val_predictions = knn_classifier.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Sample predictions:", val_predictions[:5])

test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)
print(test_labels[1:10], test_img_paths[1:10])
test_predictions = knn_classifier.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
distances, indices = knn_classifier.kneighbors(test_features, n_neighbors=5) # Get the nearest neighbors for each test image

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Sample test predictions:", test_predictions[:5])

#------------------------------------------------------------------------------------------------------
# Generate the confusion matrix
#------------------------------------------------------------------------------------------------------
cm = confusion_matrix(test_labels, test_predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('plots/cm_knn2.png')

#------------------------------------------------------------------------------------------------------
# Top 5 most confidently correct predictions
#------------------------------------------------------------------------------------------------------

# Calculate the sum of distances for each test sample
cumulative_distances = np.sum(distances, axis=1)

# Get indices of the 5 most correct predictions based on smallest cumulative distances
top_5_indices = np.argsort(cumulative_distances)[:5]

print("Top 5 most correct predictions based on smallest cumulative distances:")
for idx in top_5_indices:
    print(f"Test Image Path: {test_img_paths[idx]}")
    print(f"Predicted Label: {test_predictions[idx]}")
    print(f"Actual Label: {test_labels[idx]}")
    print(f"Distances: {distances[idx]}")
    # Get the paths of the nearest neighbors
    neighbor_paths = [train_img_paths[i] for i in indices[idx]]
    print(f"Paths of nearest neighbors: {neighbor_paths}")
    print()  # for better readability

#------------------------------------------------------------------------------------------------------
# Visualise the top 5 most confidently correct predictions 
#------------------------------------------------------------------------------------------------------
label_map = {0: 'Accessories', 1: 'Inscriptions', 2: 'Tools'}  
# Function to load and display images
def load_and_display_image(img_path, ax, title):
    image = Image.open(img_path)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

# Display the top 5 most confident predictions along with their nearest neighbors
fig, axes = plt.subplots(5, 6, figsize=(15, 15))  # 5 rows, 6 columns (1 test image + 5 neighbors per row)

for row, idx in enumerate(top_5_indices):
    test_image_path = test_img_paths[idx]
    predicted_label = label_map[test_predictions[idx]]  # Map numeric label to name
    actual_label = label_map[test_labels[idx]]  # Map numeric label to name
    distances_to_neighbors = distances[idx]
    neighbor_indices = indices[idx]
    neighbor_paths = [train_img_paths[i] for i in neighbor_indices]

    # Display the test image
    load_and_display_image(test_image_path, axes[row, 0], f'Test Image\nPred: {predicted_label}\nActual: {actual_label}')

    # Display the 5 nearest neighbors
    for col, neighbor_idx in enumerate(neighbor_indices):
        neighbor_path = train_img_paths[neighbor_idx]
        neighbor_label = label_map[train_labels[neighbor_idx]]  # Map numeric label to name
        distance = distances_to_neighbors[col]
        load_and_display_image(neighbor_path, axes[row, col + 1], f'Neighbor {col + 1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')


plt.tight_layout()
plt.savefig('plots/knnsimilarity3.png')

#------------------------------------------------------------------------------------------------------
# Top 5 most confidently incorrect predictions
#------------------------------------------------------------------------------------------------------
# Identify incorrectly predicted samples
incorrect_indices = np.where(test_predictions != test_labels)[0]

# Sort the incorrectly predicted samples by cumulative distances
sorted_incorrect_indices = incorrect_indices[np.argsort(cumulative_distances[incorrect_indices])]

# Get the top 5 most confidently incorrect predictions
top_5_incorrect_indices = sorted_incorrect_indices[:5]

# Display the top 5 most confidently incorrect predictions along with paths of their nearest neighbors
print("Top 5 most confidently incorrect predictions based on smallest cumulative distances:")

for idx in top_5_incorrect_indices:
    print(f"Test Image Path: {test_img_paths[idx]}")
    print(f"Predicted Label: {test_predictions[idx]}")
    print(f"Actual Label: {test_labels[idx]}")
    print(f"Distances: {distances[idx]}")
    # Get the paths of the nearest neighbors
    neighbor_paths = [train_img_paths[i] for i in indices[idx]]
    print(f"Paths of nearest neighbors: {neighbor_paths}")
    print()  # for better readability


fig, axes = plt.subplots(5, 6, figsize=(15, 15))  # 5 rows, 6 columns (1 test image + 5 neighbors per row)

for row, idx in enumerate(top_5_incorrect_indices):
    test_image_path = test_img_paths[idx]
    predicted_label = label_map[test_predictions[idx]]  # Map numeric label to name
    actual_label = label_map[test_labels[idx]]  # Map numeric label to name
    distances_to_neighbors = distances[idx]
    neighbor_indices = indices[idx]
    neighbor_paths = [train_img_paths[i] for i in neighbor_indices]

    # Display the test image
    load_and_display_image(test_image_path, axes[row, 0], f'Test Image\nPred: {predicted_label}\nActual: {actual_label}')

    # Display the 5 nearest neighbors
    for col, neighbor_idx in enumerate(neighbor_indices):
        neighbor_path = train_img_paths[neighbor_idx]
        neighbor_label = label_map[train_labels[neighbor_idx]]  # Map numeric label to name
        distance = distances_to_neighbors[col]
        load_and_display_image(neighbor_path, axes[row, col + 1], f'Neighbor {col + 1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')

plt.tight_layout()
plt.savefig('plots/knnincorrect.png')

#save all the paths to log files 