#--------------------------------------------------------------------------------------------------------------------------
# Packages needed
#--------------------------------------------------------------------------------------------------------------------------
from torchvision.models.resnet import resnet50
from dataloader import CustomImageDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from models import ResNet, ResidualBlock
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
#--------------------------------------------------------------------------------------------------------------------------
# Generating dataset
#--------------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = '/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 2'
dataset = CustomImageDataset(root_dir)

train_size = int(0.6 * len(dataset))  # 60% of data for training
val_size = int(0.2 * len(dataset))    # 20% of data for validation
test_size = len(dataset) - train_size - val_size  # Remaining data for testing

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
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

#--------------------------------------------------------------------------------------------------------------------------
# Training Loop for ResNet from Scratch
#--------------------------------------------------------------------------------------------------------------------------
# num_classes = 10
# num_epochs = 10
# batch_size = 16
# learning_rate = 0.01

# model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

# total_step = len(train_loader)

# for epoch in tqdm(range(num_epochs)):
#     for i, (images, labels) in enumerate(train_loader):
#         # Move tensors to the configured device
#         images = images.to(device)
#         labels = labels.to(device)


#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)


#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         del images, labels, outputs
#         torch.cuda.empty_cache()
#         gc.collect()

#     print ('Epoch [{}/{}], Loss: {:.4f}'
#                    .format(epoch+1, num_epochs, loss.item()))

#     # Validation
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in valid_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             del images, labels, outputs

#         print('Accuracy of the network on the {} validation images: {} %'.format(130, 100 * correct / total))

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         del images, labels, outputs

#     print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))


#--------------------------------------------------------------------------------------------------------------------------
# Pretrained ResNet 50
#--------------------------------------------------------------------------------------------------------------------------
resnet50 = resnet50(pretrained=True)

num_features = resnet50.fc.in_features
num_classes = 2
resnet50.fc = nn.Linear(num_features, num_classes)

# Load model to device
model = resnet50.to(device)

def train(model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
    total_step = len(train_loader)

    for epoch in tqdm(range(args.num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, args.num_epochs, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(126, 100 * correct / total))

train(model, args)
