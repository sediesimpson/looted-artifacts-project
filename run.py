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
import time
#--------------------------------------------------------------------------------------------------------------------------
# Generating dataset
#--------------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = '/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 3'
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
train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

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
    # Generate a timestamp to include in the log file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "train_val_logs"
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
    log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
    total_step = len(train_loader)

    train_losses = []
    valid_losses = []


    with open(log_file, 'w') as f:
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0

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

            # Keeping track of loss
                running_loss += loss.item()
                epoch_train_loss = running_loss / len(train_loader)
                train_losses.append(epoch_train_loss)
            # del images, labels, outputs
            # torch.cuda.empty_cache()
            # gc.collect()
                print('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_train_loss))

        #print ('Epoch [{}/{}], Loss: {:.4f}'
                       #.format(epoch+1, args.num_epochs, loss.item()))

            model.eval()  # Set model to evaluation mode
            valid_loss = 0.0
            correct = 0
            total = 0

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

            epoch_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(epoch_valid_loss)
               # del images, labels, outputs
            accuracy = 100 * correct / total
            print('Validation Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch_valid_loss, accuracy))
            #print('Accuracy of the network on the {} validation images: {} %'.format(126, 100 * correct / total))

            # Write to log file
            f.write('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Accuracy: {:.2f}%\n'
                                .format(epoch + 1, args.num_epochs, epoch_train_loss, epoch_valid_loss, accuracy))
        model.eval()
        test_loss = 0
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
                    #del images, labels, outputs
        print('Accuracy of the network on the {} test images: {} %'.format(378, 100 * correct / total))

        final_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
            # Append final test loss to the log file
    with open(log_file, 'a') as f:
        f.write('\nFinal Test Loss: {:.4f}, Test Accuracy: {:.2f}%\n'.format(final_test_loss, test_accuracy))

train(model, args)
