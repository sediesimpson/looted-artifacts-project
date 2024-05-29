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
from sklearn.metrics import confusion_matrix
import pandas as pd
from finaldataloader import CustomImageDataset
import argparse
import torch.nn.functional as F
#--------------------------------------------------------------------------------------------------------------------------
# Define argsparser
#--------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description = 'Running Baseline Models')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--root_dir', type=str, default="/rds/user/sms227/hpc-work/dissertation/data/Test10classes")
parser.add_argument('--validation_split', type=int, default=0.1)
parser.add_argument('--test_split', type=int, default=0.1)
parser.add_argument('--shuffle_dataset', type=bool, default=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=10)
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

#--------------------------------------------------------------------------------------------------------------------------
# Training, Validation and Testing Functions
#--------------------------------------------------------------------------------------------------------------------------

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_step = len(train_loader)
    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # Returns the average loss (this will be after every epoch)
    average_loss = running_loss / total_step
    print('Training Loss: {:.4f}'.format(average_loss))
    torch.save(model.state_dict(), 'model_state_dict.pth')
    print("Saved trained model.")
    return average_loss

def validate(model, valid_loader, criterion, device):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_loss /= len(valid_loader)
    accuracy = 100.0 * correct / total
    print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(validation_loss, accuracy))
    return validation_loss, accuracy

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    misclassified_paths = []
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

             # Calculate softmax probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Print or log softmax probabilities if needed
            #print(probabilities)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Collect misclassified images
            misclassified_idx = (predicted != labels).cpu().numpy().astype(bool)
            misclassified_images.extend(images[misclassified_idx].cpu())
            misclassified_labels.extend(labels[misclassified_idx].cpu().numpy())
            misclassified_predictions.extend(predicted[misclassified_idx].cpu().numpy())
            misclassified_paths.extend([paths[i] for i in range(len(paths)) if misclassified_idx[i]])


    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    unique_labels = np.unique(all_labels)
    class_names = [str(label) for label in unique_labels]
    # Convert to DataFrame for better visual
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    # Add totals
    cm_df['Total True'] = cm_df.sum(axis=1)
    total_pred = cm_df.sum(axis=0)
    total_pred.name = 'Total Predicted'
    cm_df = pd.concat([cm_df, total_pred.to_frame().T])
    # Print confusion matrix as table
    print("\nConfusion Matrix:")
    print(cm_df)

    return accuracy, cm_df, misclassified_images, misclassified_labels, misclassified_predictions, misclassified_paths

#--------------------------------------------------------------------------------------------------------------------------
# Function to visualise misclassified images 
#--------------------------------------------------------------------------------------------------------------------------
def visualise_misclassified(paths, true_labels, predicted_labels, max_images=5):
    num_misclassified = len(paths)
    if num_misclassified > 0:
        fig, axes = plt.subplots(1, min(num_misclassified, max_images), figsize=(15, 3))
        #fig.suptitle('Misclassified Images')
        
        for i, ax in enumerate(axes):
            if i >= num_misclassified:
                break
            image = Image.open(paths[i])  # Open the original image
            ax.imshow(image)
            ax.set_title(f'True: {true_labels[i]} Pred: {predicted_labels[i]}')
            ax.axis('off')
        plt.savefig('plots/misclassified_10.png')
    else:
        print("No misclassified images to display.")

#--------------------------------------------------------------------------------------------------------------------------
# Function to visualise top N probabilities for an image and class 
#--------------------------------------------------------------------------------------------------------------------------

def visualise_top_probabilities(model, image, device, class_names, k=3):
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k)

    # Convert to CPU and numpy for easy handling
    top_probs = top_probs.cpu().numpy().flatten()
    top_classes = top_classes.cpu().numpy().flatten()

    # Convert class indices to class names
    labels = [class_names[idx] for idx in top_classes]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.bar(labels, top_probs, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Top Probabilities and Associated Classes')
    plt.ylim([0, 1])  # Since probability cannot exceed 1
    plt.savefig('plots/top_probabilities.png')

#--------------------------------------------------------------------------------------------------------------------------
# Running the model
#--------------------------------------------------------------------------------------------------------------------------
from modelcomplete import CustomResNet50, CustomClassifier
# num_classes = 3
# hidden_features = 64
# learning_rate = 0.001
# num_epochs = 100

num_classes = args.num_classes
hidden_features = args.hidden_features
learning_rate = args.lr
num_epochs = args.num_epochs 

model = CustomResNet50(num_classes, hidden_features)

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
print(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.custom_classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Generate a timestamp to include in the log file name
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = "train_val_logs"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

with open(log_file, 'a') as log:
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
        scheduler.step()  # Step the learning rate scheduler
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')

    test_accuracy, confusion_matrix_df, misclassified_images, misclassified_labels, misclassified_predictions, misclassified_paths = test(model, test_loader, device)
    log.write(f'Test Accuracy: {test_accuracy:.2f}%\n')
    

    visualise_misclassified(misclassified_paths, misclassified_labels, misclassified_predictions, max_images=5)
    visualise_top_probabilities(model, image, device, class_names, k=3)
