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
from old.finaldataloader import CustomImageDataset
import argparse
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from old.finaldataloader import *

# def preprocess_image(image):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = preprocess(image)
#     return image

def generate_saliency_map(model, input_image, predicted_class):
    input_image.requires_grad_()
    output = model(input_image)    # Forward pass
    model.zero_grad()  # Zero all existing gradients
    output[0, predicted_class].backward()  # Backward pass to get gradients
    saliency = input_image.grad.data.abs().squeeze().sum(dim=0)  # Get the gradients of the input image
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())  # Normalise the saliency map
    return saliency.cpu().numpy()

# function to overlay saliency map onto image; include colour map 
def overlay_saliency_on_image(image, saliency_map, alpha=0.5, cmap='jet'):
    image = image.convert("RGB")  # Convert the image to RGB 
    image_np = np.array(image)

    saliency_map_resized = Image.fromarray((saliency_map * 255).astype(np.uint8))     # Apply color map to the saliency map
    saliency_map_resized = saliency_map_resized.resize(image.size, resample=Image.BILINEAR)
    saliency_map_resized = np.array(saliency_map_resized)

    saliency_map_colored = plt.get_cmap(cmap)(saliency_map_resized / 255.0)[:, :, :3]     # Apply color map
    saliency_map_colored = (saliency_map_colored * 255).astype(np.uint8)

    overlay = image_np * (1 - alpha) + saliency_map_colored * alpha     # Combine the image and the saliency map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)  # Ensure the overlay is in the range [0, 255] and convert to uint8
    return Image.fromarray(overlay)

def plot_saliency_maps(saliency_maps, output_dir="saliency_maps", alpha=0.5, cmap='jet'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image_path, saliency_map, predicted_class, true_label) in enumerate(saliency_maps):
        image = Image.open(image_path)
        overlay_image = overlay_saliency_on_image(image, saliency_map, alpha, cmap)
    
        plt.figure(figsize=(10, 5))
        plt.imshow(overlay_image)  # Plot the original image with saliency map overlay
        plt.title(f"True: {true_label}, Predicted: {predicted_class}")

        mappable = plt.cm.ScalarMappable(cmap=cmap)   # Add a color bar
        mappable.set_array(saliency_map)
        plt.colorbar(mappable, label="Saliency Intensity")

        plt.savefig(os.path.join(output_dir, f"saliency_map_overlay_{idx}.png"))
        plt.close()

def extract_image_paths(loader, dataset):
    paths_list = []
    for indices in loader.sampler:
        paths_list.append(dataset.img_paths[indices])
    return paths_list

def validate_image_paths(train_loader, valid_loader, test_loader, dataset):
    train_paths = extract_image_paths(train_loader, dataset)
    valid_paths = extract_image_paths(valid_loader, dataset)
    test_paths = extract_image_paths(test_loader, dataset)

    # Convert lists to sets for comparison
    train_set = set(train_paths)
    valid_set = set(valid_paths)
    test_set = set(test_paths)

    # Check for overlaps
    train_val_overlap = train_set & valid_set
    train_test_overlap = train_set & test_set
    valid_test_overlap = valid_set & test_set

    if train_val_overlap:
        print("Overlap found between train and validation sets.")
        print("Overlap paths:", train_val_overlap)
    else:
        print("No overlap between train and validation sets.")

    if train_test_overlap:
        print("Overlap found between train and test sets.")
        print("Overlap paths:", train_test_overlap)
    else:
        print("No overlap between train and test sets.")

    if valid_test_overlap:
        print("Overlap found between validation and test sets.")
        print("Overlap paths:", valid_test_overlap)
    else:
        print("No overlap between validation and test sets.")

# #--------------------------------------------------------------------------------------------------------------------------
# # Function to visualise misclassified images 
# #--------------------------------------------------------------------------------------------------------------------------
# def visualise_misclassified(paths, true_labels, predicted_labels, max_images=5):
#     num_misclassified = len(paths)
#     if num_misclassified > 0:
#         fig, axes = plt.subplots(1, min(num_misclassified, max_images), figsize=(15, 3))
#         #fig.suptitle('Misclassified Images')
        
#         for i, ax in enumerate(axes):
#             if i >= num_misclassified:
#                 break
#             image = Image.open(paths[i])  # Open the original image
#             ax.imshow(image)
#             ax.set_title(f'True: {true_labels[i]} Pred: {predicted_labels[i]}')
#             ax.axis('off')
#         plt.savefig('plots/misclassified_120624.png')
#     else:
#         print("No misclassified images to display.")

# #--------------------------------------------------------------------------------------------------------------------------
# # Function to visualise top N predictions and probabilities
# #--------------------------------------------------------------------------------------------------------------------------
# def get_top_n_subset(indices, top_n_predictions, top_n_probabilities):
#     subset_top_n_predictions = [top_n_predictions[i] for i in indices]
#     subset_top_n_probabilities = [top_n_probabilities[i] for i in indices]
#     return subset_top_n_predictions, subset_top_n_probabilities