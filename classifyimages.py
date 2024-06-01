#-----------------------------------------------------------------------------------------------------------------------------
# Import packages
#-----------------------------------------------------------------------------------------------------------------------------
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
from modelcomplete import CustomResNet50, CustomClassifier
#-----------------------------------------------------------------------------------------------------------------------------
# Function to classify new image based off of trained model 
#-----------------------------------------------------------------------------------------------------------------------------
def classify_image(model, image_path, device, class_names, top_n=3):
    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move the image tensor to the appropriate device
    image_tensor = image_tensor.to(device)

    # Set the model to evaluation mode and disable gradient calculation
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image_tensor)
        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate softmax probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the top N predictions and their probabilities
        top_n_probabilities, top_n_predictions = torch.topk(probabilities, top_n, dim=1)

    # Convert predictions and probabilities to numpy arrays for easier handling
    top_n_predictions = top_n_predictions.cpu().numpy()[0]
    top_n_probabilities = top_n_probabilities.cpu().numpy()[0]

    # Get the predicted label (the class with the highest probability)
    predicted_label = top_n_predictions[0]
    confidence = top_n_probabilities[0]

    # Map the top N predictions to class names
    top_n_class_names = [class_names[i] for i in top_n_predictions]

    # Print or return the results
    print(f"Predicted Label: {predicted_label} ({class_names[predicted_label]})")
    print(f"Confidence: {confidence:.4f}")
    print("Top N Predictions and Probabilities:")
    for i in range(top_n):
        print(f"Class {top_n_predictions[i]} ({top_n_class_names[i]}): {top_n_probabilities[i]:.4f}")

    return {
        "predicted_label": predicted_label,
        "predicted_class_name": class_names[predicted_label],
        "confidence": confidence,
        "top_n_predictions": top_n_predictions,
        "top_n_class_names": top_n_class_names,
        "top_n_probabilities": top_n_probabilities,
        "processing_time": processing_time,
        "raw_outputs": outputs.cpu().numpy()  # Raw model outputs (logits)
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomResNet50(num_classes, hidden_features)
model = model.to(device)
model.load_state_dict(torch.load('models/best_model.pth'))  # Load the best model weights

class_names = [...]  # List of class names corresponding to your model's output classes
image_path = '/rds/user/sms227/hpc-work/dissertation/data/TD4Q/Accessories - Query/Berge, Paris, 10-10-2017, Lot 20.1.jpg'

result = classify_image(model, image_path, device, class_names)

# Access the returned information
predicted_label = result["predicted_label"]
predicted_class_name = result["predicted_class_name"]
confidence = result["confidence"]
top_n_predictions = result["top_n_predictions"]
top_n_class_names = result["top_n_class_names"]
top_n_probabilities = result["top_n_probabilities"]
processing_time = result["processing_time"]
raw_outputs = result["raw_outputs"]

print(f"Processing Time: {processing_time:.4f} seconds")
