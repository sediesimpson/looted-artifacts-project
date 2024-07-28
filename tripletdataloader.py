import os
from collections import defaultdict
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tripletlossmodel import CustomResNetEmbedding
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
#--------------------------------------------------------------------------------------------------------------------------
# Dataloader 
#--------------------------------------------------------------------------------------------------------------------------
class CustomImageDatasetDup(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels = self.get_image_paths_and_labels()
        self.transform = ResNet50_Weights.DEFAULT.transforms()
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.classes = list(self.label_to_idx.keys())

        # Create a dictionary to store image paths by class
        self.class_to_img_paths = defaultdict(list)
        for img_path, label in zip(self.img_paths, self.labels):
            self.class_to_img_paths[label].append(img_path)

    def get_image_paths_and_labels(self):
        img_paths = []
        labels = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                    full_path = os.path.join(root, file)
                    img_paths.append(full_path)
                    label = os.path.basename(root)
                    labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        anchor_path = self.img_paths[idx]
        anchor_label = self.labels[idx]

        # Select a positive example from the same class
        positive_path = random.choice(self.class_to_img_paths[anchor_label])
        while positive_path == anchor_path:
            positive_path = random.choice(self.class_to_img_paths[anchor_label])

        # Select a negative example from a different class
        negative_label = random.choice([label for label in self.classes if label != anchor_label])
        negative_path = random.choice(self.class_to_img_paths[negative_label])

        # Load images
        anchor_image = self.load_image(anchor_path)
        positive_image = self.load_image(positive_path)
        negative_image = self.load_image(negative_path)

        return anchor_image, positive_image, negative_image, anchor_path, positive_path, negative_path

    def load_image(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        if self.transform:
            image = self.transform(image)
        return image

    def count_images_per_label(self):
        # Method to count the number of images for each label
        label_counts = {cls_name: 0 for cls_name in self.classes}
        for label in self.labels:
            label_counts[label] += 1
        return label_counts

    def count_unique_labels(self):
        # Method to count the number of unique labels
        unique_labels = set(self.labels)
        return len(unique_labels)
    

#--------------------------------------------------------------------------------------------------------------------------
# Building Dataset
#--------------------------------------------------------------------------------------------------------------------------
root_dir = '/rds/user/sms227/hpc-work/dissertation/data/duplicatedata'
triplet_dataset = CustomImageDatasetDup(root_dir)
triplet_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=32, shuffle=True)

#--------------------------------------------------------------------------------------------------------------------------
# Splitting into train, test, validate
#--------------------------------------------------------------------------------------------------------------------------
def split_dataset(dataset, val_split=0.2, test_split=0.1):
    train_idx, temp_idx = train_test_split(
        list(range(len(dataset))), test_size=(val_split + test_split)
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_split / (val_split + test_split))
    )
    return train_idx, val_idx, test_idx

# Create subsets
train_idx, val_idx, test_idx = split_dataset(triplet_dataset)
train_subset = Subset(triplet_dataset, train_idx)
val_subset = Subset(triplet_dataset, val_idx)
test_subset = Subset(triplet_dataset, test_idx)

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

#--------------------------------------------------------------------------------------------------------------------------
# Train Loop
#--------------------------------------------------------------------------------------------------------------------------
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0.0
    for i, (anchor, positive, negative, _, _, _) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = criterion(anchor_output, positive_output, negative_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}], Train Loss: {avg_epoch_loss:.4f}")


#--------------------------------------------------------------------------------------------------------------------------
# Validate Loop
#--------------------------------------------------------------------------------------------------------------------------
def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, (anchor, positive, negative, _, _, _) in enumerate(tqdm(val_loader, desc="Validating")):
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = criterion(anchor_output, positive_output, negative_output)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

#--------------------------------------------------------------------------------------------------------------------------
# Test Loop
#--------------------------------------------------------------------------------------------------------------------------
# Test function
def test(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for i, (anchor, positive, negative, _, _, _) in enumerate(tqdm(test_loader, desc="Testing")):
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = criterion(anchor_output, positive_output, negative_output)
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

#--------------------------------------------------------------------------------------------------------------------------
# Create model
#--------------------------------------------------------------------------------------------------------------------------
embedding_dim = 128 
weights = models.ResNet50_Weights.DEFAULT  
model = CustomResNetEmbedding(embedding_dim, weights=weights)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, triplet_loss, epoch)
    validate(model, val_loader, triplet_loss)

# Test the model
test(model, test_loader, triplet_loss)