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
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from finaldataloader import *

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image

# def generate_saliency_map(model, input_image, predicted_class):
#     input_image.requires_grad_()

#     # Forward pass
#     output = model(input_image)

#     # Zero all existing gradients
#     model.zero_grad()

#     # Backward pass to get gradients
#     output[0, predicted_class].backward()

#     # Get the gradients of the input image
#     saliency = input_image.grad.data.abs().squeeze().sum(dim=0)


#     # Normalize the saliency map
#     saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

#     return saliency.cpu().numpy()

def generate_saliency_map(model, input_image, predicted_class):
    input_image.requires_grad_()

    # Forward pass
    output = model(input_image)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to get gradients
    output[0, predicted_class].backward()

    # Get the gradients of the input image
    saliency = input_image.grad.data.abs().squeeze().sum(dim=0)

    # Normalize the saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency.cpu().numpy()


# def plot_and_save_saliency_maps(saliency_maps, output_dir="saliency_maps"):
#     import os
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for idx, (image_path, saliency_map, predicted_class, true_label) in enumerate(saliency_maps):
#         plt.figure(figsize=(10, 5))

#         # Plot the original image
#         plt.subplot(1, 2, 1)
#         image = Image.open(image_path)
#         plt.imshow(image)
#         plt.title(f"True: {true_label}, Predicted: {predicted_class}")

#         # Plot the saliency map
#         plt.subplot(1, 2, 2)
#         plt.imshow(saliency_map, cmap='hot')
#         plt.title("Saliency Map")

#         # Save the figure
#         plt.savefig(os.path.join(output_dir, f"saliency_map_{idx}.png"))
#         plt.close()  # Close the figure to free up memory

def overlay_saliency_on_image(image, saliency_map, alpha=0.5, cmap='jet'):
    """
    Overlay a saliency map on an image with a color map.
    Args:
        image (PIL.Image): The original image.
        saliency_map (numpy.ndarray): The saliency map.
        alpha (float): The transparency for the saliency map.
        cmap (str): The color map to use for the saliency map.
    Returns:
        PIL.Image: The image with the saliency map overlay.
    """
    # Convert the image to RGB if it is not already
    image = image.convert("RGB")
    image_np = np.array(image)

    # Apply color map to the saliency map
    saliency_map_resized = Image.fromarray((saliency_map * 255).astype(np.uint8))
    saliency_map_resized = saliency_map_resized.resize(image.size, resample=Image.BILINEAR)
    saliency_map_resized = np.array(saliency_map_resized)

    # Apply color map
    saliency_map_colored = plt.get_cmap(cmap)(saliency_map_resized / 255.0)[:, :, :3]
    saliency_map_colored = (saliency_map_colored * 255).astype(np.uint8)

    # Combine the image and the saliency map
    overlay = image_np * (1 - alpha) + saliency_map_colored * alpha

    # Ensure the overlay is in the range [0, 255] and convert to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)

def plot_and_save_saliency_maps(saliency_maps, output_dir="saliency_maps", alpha=0.5, cmap='jet'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image_path, saliency_map, predicted_class, true_label) in enumerate(saliency_maps):
        image = Image.open(image_path)
        overlay_image = overlay_saliency_on_image(image, saliency_map, alpha, cmap)

        # Plot and save the figure with the color bar
        plt.figure(figsize=(10, 5))

        # Plot the original image with saliency map overlay
        plt.imshow(overlay_image)
        plt.title(f"True: {true_label}, Predicted: {predicted_class}")

        # Add a color bar
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(saliency_map)
        plt.colorbar(mappable, label="Saliency Intensity")

        # Save the figure
        plt.savefig(os.path.join(output_dir, f"saliency_map_overlay_{idx}.png"))
        plt.close()

