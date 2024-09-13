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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
from classdataloader import *
import argparse
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import sys
from classificationmodel import CustomResNetClassifier
from tqdm import tqdm
#--------------------------------------------------------------------------------------------------------------------------
# Define argsparser
#--------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description = 'Running Baseline Models')

parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--root_dir', type=str, default="/rds/user/sms227/hpc-work/dissertation/data/la_data")
parser.add_argument('--validation_split', type=int, default=0.1)
parser.add_argument('--test_split', type=int, default=0.1)
parser.add_argument('--shuffle_dataset', type=bool, default=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=28)
parser.add_argument('--hidden_features', type=int, default=512)
args = parser.parse_args()

#--------------------------------------------------------------------------------------------------------------------------
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
root_dir = args.root_dir
batch_size = args.batch_size
validation_split = args.validation_split
shuffle_dataset = args.shuffle_dataset
random_seed = args.random_seed
test_split = args.test_split

# Create dataset
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"
dataset = CustomImageDataset2(root_dir)

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
#--------------------------------------------------------------------------------------------------------------------------
# Function to save a checkpoint
#--------------------------------------------------------------------------------------------------------------------------
def save_checkpoint(epoch, model, optimizer, loss, best_model, folder="checkpoints", filename="model.pt"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

    # If this is the best model, save a separate checkpoint
    if best_model:
        best_path = os.path.join(folder, "best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, best_path)
#--------------------------------------------------------------------------------------------------------------------------
# Training, Validation and Testing Functions
#--------------------------------------------------------------------------------------------------------------------------
# class_names = ['Accessories','Altars','Candelabra','Coins - Metals','Columns - Capitals',
# 'Decorative Tiles','Egyptian Coffins','Figurines','Fossils','Frescoes - Mosaics'
# ,'Heads','Human Parts','Inscriptions','Islamic','Jewelry','Manuscripts','Mirrors'
# ,'Musical Instruments','Oscilla','Other Objects','Reliefs','Sarcophagi - Urns',
# 'Sardinian Boats','Seal Stones - Seals - Stamps','Statues','Tools','Vases','Weapons']

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total = 0 
    print('Device in training loop:', device)

    for i, (images, labels, _) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device).float()  # float labels for BCEWithLogitsLoss
        # Forward pass
        outputs = model(images)
        # Compute the loss using BCEWithLogitsLoss 
        loss = criterion(outputs, labels)
        # Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  # Accumulate loss 


    average_loss = running_loss / len(train_loader)  # Compute the average loss for the epoch
    print('Training Loss: {:.4f}'.format(average_loss))
    return average_loss


def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_step = len(valid_loader)
    total = 0
    print('Device in validation loop:', device)
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device).float()  # Float labels for BCEWithLogitsLoss

            outputs = model(images)

            loss = criterion(outputs, labels) 

            running_loss += loss.item() 
            probs = torch.sigmoid(outputs) 
            predicted = probs.round()

            correct_predictions += (predicted == labels).sum().item()

            total += labels.size(1) * batch_size

    
    # Compute the average loss for the epoch
    val_loss = running_loss / len(valid_loader) 
    # Compute the accuracy
    val_accuracy = correct_predictions / total 
    
    print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_accuracy))
    
    return val_loss, val_accuracy

def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    
    print('Device in test loop:', device)
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            predicted = probs.round()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='samples')
    recall = recall_score(all_labels, all_predictions, average='samples')
    f1 = f1_score(all_labels, all_predictions, average='samples')
    
    # Calculate per-class accuracy
    per_class_accuracy = (all_predictions == all_labels).sum(axis=0) / all_labels.shape[0]

    # Generate normalized confusion matrix
    cm = confusion_matrix(all_labels.argmax(axis=1), all_predictions.argmax(axis=1), labels=range(28), normalize='true')

    # Print results
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(accuracy, precision, recall, f1))
    
    # Print per-class accuracy
    for i, acc in enumerate(per_class_accuracy):
        print(f'Accuracy for class {i}: {acc:.4f}')
    
    return accuracy, precision, recall, f1, per_class_accuracy, cm
#--------------------------------------------------------------------------------------------------------------------------
# Running the model
#--------------------------------------------------------------------------------------------------------------------------
num_classes = args.num_classes
hidden_features = args.hidden_features
learning_rate = args.lr
num_epochs = args.num_epochs 

print("\nVariables Used:\n")
print(f'Number of Epochs: {args.num_epochs}\n')
print(f'Number of Classes: {args.num_classes}\n')
print(f'Hidden Features: {args.hidden_features}\n')
print(f'Learning Rate: {args.lr}\n')

hidden_dim = 256  # Intermediate hidden layer size
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = CustomResNetClassifier(hidden_dim, num_classes, weights=weights)

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
print(device)
#--------------------------------------------------------------------------------------------------------------------------
# Weighting function
#--------------------------------------------------------------------------------------------------------------------------
# Get image paths and labels
img_paths, labels = dataset.get_image_paths_and_labels()

# Convert list of labels to a numpy array for easier manipulation
labels_array = np.array(labels)

# Calculate pos_weight for each class
positive_counts = np.sum(labels_array, axis=0)

negative_counts = labels_array.shape[0] - positive_counts
pos_weight = negative_counts / positive_counts

# Convert pos_weight to a tensor
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
# print('pos_weight_tensor:', pos_weight_tensor)


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
criterion = criterion.to(device)
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


best_val_accuracy = 0.0 

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
    print('-' * 30)

    # Check if the current model is the best
    best_model = val_accuracy > best_val_accuracy
    if best_model:
        best_val_accuracy = val_accuracy

    save_checkpoint(epoch, model, optimizer, val_loss, best_model, folder="checkpoints", filename="model_epoch_{}.pt".format(epoch))


accuracy, precision, recall, f1, per_class_accuracy, cm = test(model, test_loader, device)
print('-' * 30)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(28), yticklabels=range(28))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('plots/cm_dup.png')
