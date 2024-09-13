#--------------------------------------------------------------------------------------------------------------------------
# Packages needed
#--------------------------------------------------------------------------------------------------------------------------
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
import torchvision.models as models
from dataloader import CustomImageDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from old.models import ResNet, ResidualBlock
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gc
from main import args
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from dataloader import AdjustLabels
from old.modelcomplete import CustomResNet50, CustomClassifier, criterion, num_classes, hidden_features, learning_rate, num_epochs

#--------------------------------------------------------------------------------------------------------------------------
# Training, Validation and Testing Functions
#--------------------------------------------------------------------------------------------------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device):
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

        # this prints out the running loss after every 100 epoches
        # if (i+1) % 100 == 0:
        #     print('Step [{}/{}], Loss: {:.4f}'.format(i+1, total_step, running_loss / 100))
        #     running_loss = 0.0

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

def main():

    # Model, criterion, optimizer
    # resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # num_features = resnet50.fc.in_features
    # num_classes = 3
    # resnet50.fc = nn.Linear(num_features, num_classes)
    # model = resnet50.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate)

    # Create the custom model
    model = CustomResNet50(num_classes, hidden_features)

    # Move the model to the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.custom_classifier.parameters(), lr=learning_rate)

    # Data loaders
    root_dir = '/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 4'
    dataset = CustomImageDataset(root_dir)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))
   # test_size = len(dataset) - train_size - val_size

    # Create indices for the splits
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # Create SubsetRandomSampler objects for each set
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoader objects using the samplers
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)


# Generate a timestamp to include in the log file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "train_val_logs"
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
    log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

    with open(log_file, 'a') as log:
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
            log.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')

        test_accuracy = test(model, test_loader, device)
        log.write(f'Test Accuracy: {test_accuracy:.2f}%\n')


if __name__ == "__main__":
    main()
