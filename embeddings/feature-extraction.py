#--------------------------------------------------------------------------------------------------------------------------
# Packages needed
#--------------------------------------------------------------------------------------------------------------------------
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import time
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------------------------
# Building dataloader
#--------------------------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(cls for cls in os.listdir(root_dir) if cls != '.DS_Store')
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = self.get_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_image_paths(self):
        img_paths = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, cls_name)
            for root, _, files in os.walk(cls_path):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                        img_paths.append((os.path.join(root, file), cls_name))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, cls_name = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        label = self.class_to_idx[cls_name]
        return self.transform(image), label

    def get_label_info(self):
        # Method to print the labels and their corresponding indices
        label_info = {self.class_to_idx[cls_name]: cls_name for cls_name in self.classes}
        return label_info

    def count_images_per_label(self):
        # Method to count the number of images for each label
        label_counts = {cls_name: 0 for cls_name in self.classes}
        for _, cls_name in self.img_paths:
            label_counts[cls_name] += 1
        return label_counts

#--------------------------------------------------------------------------------------------------------------------------
# Check dataloader works correctly
#--------------------------------------------------------------------------------------------------------------------------
# Define paths and parameters
root_dir = '/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 4'
dataset = CustomImageDataset(root_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Get label information
label_info = dataset.get_label_info()
print("Label Information:", label_info)

# Get the number of images per label
label_counts = dataset.count_images_per_label()
print("Number of images per label:", label_counts)

def count_labels_in_loader(loader, class_to_idx):
    # Initialize label_counts using the class indices directly
    label_counts = {idx: 0 for idx in class_to_idx.values()}
    for _, labels in loader:
        for label in labels.numpy():
            if label in label_counts:
                label_counts[label] += 1
    return label_counts

# Check label distribution in each loader
data_label_counts = count_labels_in_loader(data_loader, dataset.class_to_idx)

print("Label distribution:", data_label_counts)
#--------------------------------------------------------------------------------------------------------------------------
# Calculating feature embeddings
#--------------------------------------------------------------------------------------------------------------------------
from resnetembedding import ResNet50Embedding, CustomClassifierEmbedding

hidden_features = 64
model = ResNet50Embedding(hidden_features)
model.eval()

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to extract features
def extract_features(data_loader, model):
    features = []
    labels = []
    with torch.no_grad():  # No need to track gradients
        for images, batch_labels in tqdm(data_loader):
            images = images.to(device)
            batch_features = model(images)  # Get the embeddings from the model
            features.append(batch_features.cpu())  # Store features to a list after moving to cpu
            labels.append(batch_labels)

    features = torch.cat(features, dim=0)  # Concatenate all feature tensors
    labels = torch.cat(labels, dim=0)  # Concatenate all label tensors
    return features, labels

# Extract features
features, labels = extract_features(data_loader, model)


#--------------------------------------------------------------------------------------------------------------------------
# Reduce dimensionality of the features
#--------------------------------------------------------------------------------------------------------------------------
def reduce_dimensions(features, labels, components=2):
    tsne = TSNE(n_components=components, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

# Convert PyTorch tensor to NumPy array
features_np = features.numpy() if isinstance(features, torch.Tensor) else features
labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

# Reduce dimensions
reduced_features = reduce_dimensions(features_np, labels_np)

#--------------------------------------------------------------------------------------------------------------------------
# Plot the features
#--------------------------------------------------------------------------------------------------------------------------
def plot_reduced_features(reduced_features, labels, save_path):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Select indices for each label
        indices = labels == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Label {label}', alpha=0.5)

    plt.title('2D Visualization of Feature Vectors')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(save_path)
    plt.close()

plot_reduced_features(reduced_features, labels_np, 'plots/feature_vectors.png')
