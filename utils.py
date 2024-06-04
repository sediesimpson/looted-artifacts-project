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

