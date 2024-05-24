import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
#from finaldataloader import CustomImageDataset
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
                img_tensors.append(img.to(device))
            if not img_tensors:
                continue

            imgs = torch.stack(img_tensors)
            features = model.resnet50.avgpool(model.resnet50.layer4(model.resnet50.layer3(model.resnet50.layer2(model.resnet50.layer1(model.resnet50.maxpool(model.resnet50.relu(model.resnet50.bn1(model.resnet50.conv1(imgs)))))))))
            features = torch.flatten(features, 1)
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
        foreground_image = dataset.background_subtraction(image_path)
        if foreground_image is None:
            return None, None
        img_t = dataset.transform(foreground_image).unsqueeze(0).to(device)
        features = model.resnet50.avgpool(model.resnet50.layer4(model.resnet50.layer3(model.resnet50.layer2(model.resnet50.layer1(model.resnet50.maxpool(model.resnet50.relu(model.resnet50.bn1(model.resnet50.conv1(img_t)))))))))
        features = torch.flatten(features, 1)

    # Find the label of the query image
    query_label = None
    for path, label in dataset.img_paths:
        if path == image_path:
            query_label = label
            break

    return features.cpu().numpy().squeeze(), query_label  # Remove batch dimension and convert to numpy array

#------------------------------------------------------------------------------------------------------------------------
# Define the similarity function and # Define top N similar function
#------------------------------------------------------------------------------------------------------------------------

def compute_similarities(query_features, dataset_features):
    similarities = [cosine(query_features, features) for features in dataset_features]
    return similarities

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

dataset_dir = "/rds/user/sms227/hpc-work/dissertation/data/Test Dataset 4"
dataset = CustomImageDataset(root_dir=dataset_dir)

#------------------------------------------------------------------------------------------------------------------------
# Create dataloader
#------------------------------------------------------------------------------------------------------------------------
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
num_classes = 3
hidden_features = 64
model = CustomResNet50(num_classes=num_classes, hidden_features=hidden_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Extract features for whole dataset
features, labels, img_paths = extract_features(dataloader, model, device)

# Print shape of extracted features
print("Shape of extracted features:", features.shape)
print("Shape of labels:", labels.shape)

#------------------------------------------------------------------------------------------------------------------------
# Query, similarity and top N images using cosine similarity
#------------------------------------------------------------------------------------------------------------------------
query_image_path = "/rds/user/sms227/hpc-work/dissertation/data/Test Dataset 4/Accessories/Barakat Volume-11, FZ210.JPG"
query_features, query_label = extract_query_features(query_image_path, dataset, model, device)
similarities = compute_similarities(query_features, features)
n = 4
top_n_indices = retrieve_top_n_similar(similarities, img_paths, query_image_path, n=n)
print("Top N similar image indices:", top_n_indices)
print("Top N similar image indices:", top_n_indices)

#------------------------------------------------------------------------------------------------------------------------
# Query, similarity and top N images using KNN classifier
#------------------------------------------------------------------------------------------------------------------------
# from sklearn.metrics import roc_curve, auc

# # Compute distances for a validation set
# validation_distances = []
# validation_labels = []
# for val_img_features, val_label in zip(validation_distances, validation_labels):
#     distances, _ = knn.kneighbors([val_img_features], n_neighbors=1, return_distance=True)
#     validation_distances.append(distances[0][0])
#     validation_labels.append(val_label)


# # Define true labels and predicted scores based on distances
# true_labels = [1 if val_label == query_label else 0 for val_label in validation_labels]
# predicted_scores = [-dist for dist in validation_distances]  # Use negative distance as score

# # Compute ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# # Choose the threshold with the best balance (e.g., max TPR - FPR)
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx]
# print(f'Optimal Threshold: {optimal_threshold}')

# # Define the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3, metric='cosine') # here we use cosine distance to make it comparable, but can also be euclidean distance
# # Fit the knn
# knn.fit(features, labels)

# # This is used to find the top N similar images.
# N = 3
# distances, indices = knn.kneighbors([query_features], n_neighbors=N, return_distance=True)

# # To classify it as the same object, we define a threshold
# threshold = 0.2

# # If the query image is in our database of features, it has to be below the threshold defined above. Here we check this
# if query_image_path in [img_paths[i] for i in indices[0]] and distances[0][0] < threshold:
#     print("Query image classified as the same object")
#     result_label = query_label
# else:
#     # If not same object, do prior-based classification
#     top_n_labels = labels[indices[0]]
#     prior_distribution = {label: np.sum(top_n_labels == label) / N for label in np.unique(top_n_labels)}
#     classifier_probabilities = knn.predict_proba([query_features])[0]
#     adjusted_probabilities = {label: classifier_probabilities[i] * prior_distribution.get(label, 0)
#                               for i, label in enumerate(knn.classes_)}
#     # Normalize adjusted probabilities
#     total_prob = sum(adjusted_probabilities.values())
#     normalized_probabilities = {label: prob / total_prob for label, prob in adjusted_probabilities.items()}
#     result_label = max(normalized_probabilities, key=normalized_probabilities.get)
#     print("Query image classified as:", result_label)

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
    plt.savefig('plots/top images')
visualize_similar_images(query_image_path, top_n_image_paths, top_n_labels, n=n)
