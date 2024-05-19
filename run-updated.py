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
# Define Parameters and check dataloaders
#--------------------------------------------------------------------------------------------------------------------------
# Define paths and parameters
root_dir = '/rds/user/sms227/hpc-work/dissertation/data/Test Dataset 4'
batch_size = 32
validation_split = 0.2
shuffle_dataset = True
random_seed = 42
test_split = 0.1

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
    for _, labels in loader:
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
    for i, (images, labels) in enumerate(train_loader):
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
    return average_loss

def validate(model, valid_loader, criterion, device):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
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
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))
    return accuracy

#--------------------------------------------------------------------------------------------------------------------------
# Running the model
#--------------------------------------------------------------------------------------------------------------------------
from modelcomplete import CustomResNet50, CustomClassifier
num_classes = 3
hidden_features = 64
learning_rate = 0.01
num_epochs = 20

model = CustomResNet50(num_classes, hidden_features)

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)

# # Define the loss function and the optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.custom_classifier.parameters(), lr=learning_rate)

# # Generate a timestamp to include in the log file name
# timestamp = time.strftime("%Y%m%d-%H%M%S")
# log_dir = "train_val_logs"
# os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
# log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

# with open(log_file, 'a') as log:
#     for epoch in range(num_epochs):
#         print(f'Epoch [{epoch+1}/{num_epochs}]')
#         train_loss = train(model, train_loader, criterion, optimizer, device)
#         val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
#         log.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')

#     test_accuracy = test(model, test_loader, device)
#     log.write(f'Test Accuracy: {test_accuracy:.2f}%\n')
