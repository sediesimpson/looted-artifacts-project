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

from multidataloader import *

from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

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
    
# Function to ensure no data leakage
def split_dataset(dataset, test_split=0.2, val_split=0.1, shuffle=True, random_seed=42):
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
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"
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
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# # Check for data leakage
# specific_path = "Barakat 34, GF.346.JPG"
# train_labels = set()
# val_labels = set()
# test_labels = set()

# for _, _, paths in train_loader:
#     for path in paths:
#         if os.path.basename(path) == specific_path:
#             train_labels.update(dataset.get_label_for_path(path))

# for _, _, paths in valid_loader:
#     for path in paths:
#         if os.path.basename(path) == specific_path:
#             val_labels.update(dataset.get_label_for_path(path))

# for _, _, paths in test_loader:
#     for path in paths:
#         if os.path.basename(path) == specific_path:
#             test_labels.update(dataset.get_label_for_path(path))

# print(f"Train labels for {specific_path}: {train_labels}")
# print(f"Validation labels for {specific_path}: {val_labels}")
# print(f"Test labels for {specific_path}: {test_labels}")

# def count_labels_in_loader(loader, class_to_idx):
#     # Initialize label_counts using the class indices directly
#     label_counts = {idx: 0 for idx in class_to_idx.values()}
#     for _, labels, _ in loader:
#         print(len(labels))
#         for idx, label in enumerate(labels):
#             print(label)
#             if label == 1:
#                 label_counts[idx] += 1
#     return label_counts


# # Count labels in each DataLoader
# train_label_counts = count_labels_in_loader(train_loader, dataset.class_to_idx)
# print(f"Label counts in train loader: {train_label_counts}")


# valid_label_counts = count_labels_in_loader(valid_loader, dataset.class_to_idx)
# print(f"Label counts in validation loader: {valid_label_counts}")

# test_label_counts = count_labels_in_loader(test_loader, dataset.class_to_idx)
# print(f"Label counts in test loader: {test_label_counts}")

# train_label_counts = {k: f"{v:.4f}" for k, v in train_label_counts.items()}
# valid_label_counts = {k: f"{v:.4f}" for k, v in valid_label_counts.items()}
# test_label_counts = {k: f"{v:.4f}" for k, v in test_label_counts.items()}

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

# Define a custom accuracy function for multi-label classification
def multilabel_accuracy(true_labels, pred_labels):
    sample_accuracies = np.mean(true_labels == pred_labels, axis=1)
    overall_accuracy = np.mean(sample_accuracies)
    return overall_accuracy

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 28
hidden_features = 512
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for training and validation sets
train_features, train_labels, train_img_paths = extract_features(train_loader, model, device)
val_features, val_labels, val_img_paths = extract_features(valid_loader, model, device)

# Measure the preprocessing time
start_time = time.time()
# Train the KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_classifier.fit(train_features, train_labels)

# Validate the classifier
val_predictions = knn_classifier.predict(val_features)

# Binarize predictions (if not already binary)
val_predictions = (val_predictions >= 0.5).astype(int)  # Adjust threshold if needed

# Calculate and print validation accuracy
val_accuracy = multilabel_accuracy(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
end_time  = time.time()
preprocess_time = end_time-start_time
print(preprocess_time)
sys.exit()

# Extract features for the test set
test_features, test_labels, test_img_paths = extract_features(test_loader, model, device)

# Predict on test set
test_predictions = knn_classifier.predict(test_features)

# Binarize predictions (if not already binary)
test_predictions = (test_predictions >= 0.5).astype(int)  # Adjust threshold if needed

# Calculate and print test accuracy
test_accuracy = multilabel_accuracy(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get the nearest neighbors for each test image
distances, indices = knn_classifier.kneighbors(test_features, n_neighbors=5)

print("Sample test predictions:", test_predictions[:5])

#------------------------------------------------------------------------------------------------------------------------
# # Log file generation
#------------------------------------------------------------------------------------------------------------------------
# Generate a timestamp to include in the log file name
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = "knn_logs"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

with open(log_file, 'a') as log:
    # log.write(f"Training label distribution: {train_label_counts}\n")
    # log.write(f"Validation label distribution: {valid_label_counts}\n")
    # log.write(f"Test label distribution:  {test_label_counts}\n")
    log.write(f"Batch Size:  {batch_size}\n")
    log.write(f"Validation Split:  {validation_split}\n")
    log.write(f"Test Split:  {test_split}\n")
    log.write(f"Shuffle Dataset:  {shuffle_dataset}\n")
    log.write(f"Random Seed:  {random_seed}\n")
    log.write(f"Test Split:  {test_split}\n")
    log.write(f"Number of Classes: {num_classes}\n")
    log.write(f"Hidden Features:  {hidden_features}\n")
    log.write(f"Validation Accuracy: {val_accuracy}\n")
    log.write(f"Sample predictions:  {val_predictions[:5]}\n")
    log.write(f"Test Accuracy: {test_accuracy}\n")
    log.write(f"Sample test predictions: {test_predictions[:5]}\n")

#------------------------------------------------------------------------------------------------------
# Confusion Matrices
#------------------------------------------------------------------------------------------------------
# Optionally, compute and print the confusion matrix for each label
confusion_matrices = multilabel_confusion_matrix(test_labels, test_predictions)

# Calculate accuracy per label from confusion matrices
# label_map = {0: 'Accessories', 1: 'Inscriptions', 2: 'Tools'} 
#label_map = {0: 'Figurines', 1: 'Heads', 2: 'Human Parts', 3: 'Jewelry', 4: 'Reliefs', 5: 'Seal Stones - Seals - Stamps', 6: 'Statues', 7: 'Tools', 8: 'Vases', 9: 'Weapons'}
label_map = {0: 'Accessories', 1: 'Altars', 2: 'Candelabra', 3: 'Coins - Metals', 4: 'Columns - Capitals', 5: 'Decorative Tiles', 6: 'Egyptian Coffins', 7: 'Figurines', 8: 'Fossils', 9: 'Frescoes - Mosaics', 10: 'Heads', 11: 'Human Parts', 12: 'Inscriptions', 13: 'Islamic', 14: 'Jewelry', 15: 'Manuscripts', 16: 'Mirrors', 17: 'Musical Instruments', 18: 'Oscilla', 19: 'Other Objects', 20: 'Reliefs', 21: 'Sarcophagi - Urns', 22: 'Sardinian Boats', 23: 'Seal Stones - Seals - Stamps', 24: 'Statues', 25: 'Tools', 26: 'Vases', 27: 'Weapons'}
accuracy_per_label = {}
for i, cm in enumerate(confusion_matrices):
    label_name = label_map[i]
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy_per_label[label_name] = accuracy

# Write the confusion matrix and accuracy for each label to the log file
with open(log_file, 'a') as log:
    log.write("Confusion Matrices and Accuracies Per Label:\n\n")
    
    for i, cm in enumerate(confusion_matrices):
        label_name = label_map[i]
        log.write(f"Confusion matrix for {label_name}:\n")
        log.write(f"{cm}\n")
        log.write(f"Accuracy for {label_name}: {accuracy_per_label[label_name]:.4f}\n\n")
    
    log.write("Overall accuracies per label:\n")
    for label, acc in accuracy_per_label.items():
        log.write(f"{label}: {acc:.4f}\n")
    log.write("\n")
#------------------------------------------------------------------------------------------------------
# # Top 5 most confidently correct predictions
#------------------------------------------------------------------------------------------------------
cumulative_distances = np.sum(distances, axis=1)

# Get indices of the 5 most confidently correct predictions based on smallest cumulative distances
top_5_indices = np.argsort(cumulative_distances)[:5]

with open(log_file, 'a') as log:
    log.write("Top 5 most correct predictions based on smallest cumulative distances:\n")
    for idx in top_5_indices:
        log.write(f"Test Image Path: {test_img_paths[idx]}\n")
        log.write(f"Predicted Label: {test_predictions[idx]}\n")
        log.write(f"Actual Label: {test_labels[idx]}\n")
        log.write(f"Distances: {distances[idx]}\n")
        # Get the paths of the nearest neighbors
        neighbor_paths = [train_img_paths[i] for i in indices[idx]]
        log.write(f"Paths of nearest neighbors: {neighbor_paths}\n")
        log.write("\n")  # for better readability


#------------------------------------------------------------------------------------------------------
# Visualise the top 5 most confidently correct predictions 
#------------------------------------------------------------------------------------------------------
#label_map = {0: 'Accessories', 1: 'Inscriptions', 2: 'Tools'}  
# label_map = {0: 'Figurines', 1: 'Heads', 2: 'Human Parts', 3: 'Jewelry', 4: 'Reliefs', 5: 'Seal Stones - Seals - Stamps', 6: 'Statues', 7: 'Tools', 8: 'Vases', 9: 'Weapons'}

# Define the root directory of your dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"  # Replace with the actual root directory


# Function to format labels for display
def format_labels(labels):
    return ", ".join([label_map[i] for i, label in enumerate(labels) if label == 1])

# Function to load and display images
def load_and_display_image(img_path, ax, title):
    image = Image.open(img_path)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

# Display the top 5 most confident predictions along with their nearest neighbors
fig, axes = plt.subplots(5, 6, figsize=(15, 15))  # 5 rows, 6 columns (1 test image + 5 neighbors per row)

for row, idx in enumerate(top_5_indices):
    # Construct the full path for the test image
    test_image_path = os.path.join(root_dir, test_img_paths[idx])
    predicted_label = format_labels(test_predictions[idx])  # Multi-label predictions
    actual_label = format_labels(test_labels[idx])  # Multi-label actual labels
    distances_to_neighbors = distances[idx]
    neighbor_indices = indices[idx]

    # Display the test image
    load_and_display_image(test_image_path, axes[row, 0], f'Test Image\nPred: {predicted_label}\nActual: {actual_label}')

    # Display the 5 nearest neighbors
    for col, neighbor_idx in enumerate(neighbor_indices):
        # Construct the full path for the neighbor image
        neighbor_path = os.path.join(root_dir, train_img_paths[neighbor_idx])
        neighbor_label = format_labels(train_labels[neighbor_idx])  # Multi-label neighbor labels
        distance = distances_to_neighbors[col]
        load_and_display_image(neighbor_path, axes[row, col + 1], f'Neighbor {col + 1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')

plt.tight_layout()
plt.savefig('knnplots/multicorrect_240624.png')

#------------------------------------------------------------------------------------------------------
# # Top 5 most confidently incorrect predictions
#------------------------------------------------------------------------------------------------------
# Identify incorrectly predicted samples for multi-label classification
incorrect_indices = np.where(np.any(test_predictions != test_labels, axis=1))[0]

# Sort the incorrectly predicted samples by cumulative distances
sorted_incorrect_indices = incorrect_indices[np.argsort(cumulative_distances[incorrect_indices])]

# Get the top 5 most confidently incorrect predictions
top_5_incorrect_indices = sorted_incorrect_indices[:5]

# Display the top 5 most confidently incorrect predictions along with paths of their nearest neighbors
with open(log_file, 'a') as log:
    log.write("Top 5 most confidently incorrect predictions based on smallest cumulative distances:\n")

    for idx in top_5_incorrect_indices:
        log.write(f"Test Image Path: {test_img_paths[idx]}\n")
        log.write(f"Predicted Label: {test_predictions[idx]}\n")
        log.write(f"Actual Label: {test_labels[idx]}\n")
        log.write(f"Distances: {distances[idx]}\n")
        # Get the paths of the nearest neighbors
        neighbor_paths = [train_img_paths[i] for i in indices[idx]]
        log.write(f"Paths of nearest neighbors: {neighbor_paths}\n")
        #log.write()  # for better readability

# Visualize the top 5 most confidently incorrect predictions along with paths of their nearest neighbors
fig, axes = plt.subplots(5, 6, figsize=(15, 15))  # 5 rows, 6 columns (1 test image + 5 neighbors per row)

for row, idx in enumerate(top_5_incorrect_indices):
    test_image_path = os.path.join(root_dir, test_img_paths[idx])
    predicted_label = format_labels(test_predictions[idx])  # Multi-label predictions
    actual_label = format_labels(test_labels[idx])  # Multi-label actual labels
    distances_to_neighbors = distances[idx]
    neighbor_indices = indices[idx]
    neighbor_paths = [train_img_paths[i] for i in neighbor_indices]

    # Display the test image
    load_and_display_image(test_image_path, axes[row, 0], f'Test Image\nPred: {predicted_label}\nActual: {actual_label}')

    # Display the 5 nearest neighbors
    for col, neighbor_idx in enumerate(neighbor_indices):
        neighbor_path = os.path.join(root_dir, train_img_paths[neighbor_idx])
        neighbor_label = format_labels(train_labels[neighbor_idx])  # Multi-label neighbor labels
        distance = distances_to_neighbors[col]
        load_and_display_image(neighbor_path, axes[row, col + 1], f'Neighbor {col + 1}\nLabel: {neighbor_label}\nDist: {distance:.2f}')

plt.tight_layout()
plt.savefig('knnplots/multiincorrect_240624.png')














#--------------------------------------------------------------------------------------------------------------------------------------------
# # Preprocess function
# def preprocess(image):
#     preprocess_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to the size expected by your model
#         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with the mean and std used during training
#     ])
#     image = preprocess_transforms(image)
#     return image

# # Function to extract features from a single image
# def extract_features_from_image(image_path, model, device):
#     # Load the image
#     image = Image.open(image_path)
    
#     # Preprocess the image
#     image = preprocess(image)
#     image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
#     # Extract features using the model
#     model.eval()
#     with torch.no_grad():
#         features = model.resnet50.avgpool(
#             model.resnet50.layer4(
#                 model.resnet50.layer3(
#                     model.resnet50.layer2(
#                         model.resnet50.layer1(
#                             model.resnet50.maxpool(
#                                 model.resnet50.relu(
#                                     model.resnet50.bn1(
#                                         model.resnet50.conv1(image)
#                                     )
#                                 )
#                             )
#                         )
#                     )
#                 )
#             )
#         )
#         features = torch.flatten(features, 1)
#     return features.cpu().numpy()


# # Load the new query image and extract features
# new_query_image_path = '/rds/user/sms227/hpc-work/dissertation/data/TD10Q/Confidently Incorrect CNN/Jewelry/Bonhams, London, 13-4-2011, Lot 23.2.jpg'  
# new_query_features = extract_features_from_image(new_query_image_path, model, device)

# # Predict the label for the new query image
# new_query_prediction = knn_classifier.predict(new_query_features)
# predicted_label = new_query_prediction[0]
# with open(log_file, 'a') as log:
#     log.write(f"Predicted Label for the new query image: {predicted_label}\n")

# # Get the nearest neighbors for the new query image
# distances, indices = knn_classifier.kneighbors(new_query_features, n_neighbors=2)

# # Print the paths of the nearest neighbors
# with open(log_file, 'a') as log:
#     log.write("Nearest neighbors for the new query image:\n")
#     for neighbor_idx in indices[0]:
#         neighbor_path = train_img_paths[neighbor_idx]
#         log.write(f"Path: {neighbor_path}\n")


# # Load the new query image and extract features
# new_query_image_path = '/rds/user/sms227/hpc-work/dissertation/data/TD10Q/Borderline Incorrect CNN/Figurines/Berge, Paris, 1-6-2012, Lot 49.jpg'  
# new_query_features = extract_features_from_image(new_query_image_path, model, device)

# # Predict the label for the new query image
# new_query_prediction = knn_classifier.predict(new_query_features)
# predicted_label = new_query_prediction[0]
# with open(log_file, 'a') as log:
#     log.write(f"Predicted Label for the new query image: {predicted_label}\n")

# # Get the nearest neighbors for the new query image
# distances, indices = knn_classifier.kneighbors(new_query_features, n_neighbors=2)

# # Print the paths of the nearest neighbors
# with open(log_file, 'a') as log:
#     log.write("Nearest neighbors for the new query image:\n")
#     for neighbor_idx in indices[0]:
#         neighbor_path = train_img_paths[neighbor_idx]
#         log.write(f"Path: {neighbor_path}\n")