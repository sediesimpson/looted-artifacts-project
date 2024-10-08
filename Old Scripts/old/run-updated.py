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
from old.finaldataloader import CustomImageDataset
import argparse
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from old.gradcam import *
#--------------------------------------------------------------------------------------------------------------------------
# Define argsparser
#--------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description = 'Running Baseline Models')

parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--root_dir', type=str, default="/rds/user/sms227/hpc-work/dissertation/data/TD10A")
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


train_label_counts = {k: f"{v:.4f}" for k, v in train_label_counts.items()}
valid_label_counts = {k: f"{v:.4f}" for k, v in valid_label_counts.items()}
test_label_counts = {k: f"{v:.4f}" for k, v in test_label_counts.items()}

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


def test(model, test_loader, device, top_n=3):
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_top_n_predictions = []
    all_top_n_probabilities = []
    all_paths = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    misclassified_paths = []
    correctly_classified_indices = []
    incorrectly_classified_indices = []
    confidence_correct = []
    confidence_incorrect = []
    borderline_cases = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

             # Calculate softmax probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get the top N predictions and their probabilities
            top_n_probabilities, top_n_predictions = torch.topk(probabilities, top_n, dim=1)

            # Append the top N predictions and probabilities for each image
            all_top_n_predictions.extend(top_n_predictions.cpu().numpy())
            all_top_n_probabilities.extend(top_n_probabilities.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_paths.extend(paths)

            # Collect correctly and misclassified images indices
            for j in range(len(labels)):
                confidence = probabilities[j, predicted[j]].item()
                confidence_diff = top_n_probabilities[j][0] - top_n_probabilities[j][1] if top_n > 1 else 1.0
                if predicted[j] == labels[j]:
                    correctly_classified_indices.append(len(all_labels) - len(labels) + j)
                    confidence_correct.append((confidence, len(all_labels) - len(labels) + j))
                else:
                    incorrectly_classified_indices.append(len(all_labels) - len(labels) + j)
                    confidence_incorrect.append((confidence, len(all_labels) - len(labels) + j))
                borderline_cases.append((confidence_diff, len(all_labels) - len(labels) + j, predicted[j] == labels[j]))


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
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names) # dataframe conversion
    cm_df['Total True'] = cm_df.sum(axis=1) # adding totals
    total_pred = cm_df.sum(axis=0)
    total_pred.name = 'Total Predicted'
    cm_df = pd.concat([cm_df, total_pred.to_frame().T])

    # Sort the confidences
    confidence_correct.sort(reverse=True, key=lambda x: x[0])
    confidence_incorrect.sort(key=lambda x: x[0])
    borderline_cases.sort(key=lambda x: x[0])

    # top N information
    top_n_info = {
        "all_top_n_predictions": all_top_n_predictions,
        "all_top_n_probabilities": all_top_n_probabilities,
        "correctly_classified_indices": correctly_classified_indices,
        "incorrectly_classified_indices": incorrectly_classified_indices,
        "paths": all_paths,
        "confidence_correct": confidence_correct,
        "confidence_incorrect": confidence_incorrect,
        "borderline_cases": borderline_cases,
        "all_labels": all_labels,
        "confusion_matrix": cm
    }

    return accuracy, cm_df, class_names, misclassified_images, misclassified_labels, misclassified_predictions, misclassified_paths, top_n_info

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
        plt.savefig('plots/misclassified_130624.png')
    else:
        print("No misclassified images to display.")

#--------------------------------------------------------------------------------------------------------------------------
# Function to visualise top N predictions and probabilities
#--------------------------------------------------------------------------------------------------------------------------
def get_top_n_subset(indices, top_n_predictions, top_n_probabilities):
    subset_top_n_predictions = [top_n_predictions[i] for i in indices]
    subset_top_n_probabilities = [top_n_probabilities[i] for i in indices]
    return subset_top_n_predictions, subset_top_n_probabilities

#--------------------------------------------------------------------------------------------------------------------------
# Running the model
#--------------------------------------------------------------------------------------------------------------------------
from old.modelcomplete import *
num_classes = args.num_classes
hidden_features = args.hidden_features
learning_rate = args.lr
num_epochs = args.num_epochs 

print("\nVariables Used:\n")
print(f'Number of Epochs: {args.num_epochs}\n')
print(f'Number of Classes: {args.num_classes}\n')
print(f'Hidden Features: {args.hidden_features}\n')
print(f'Learning Rate: {args.lr}\n')

model = CustomResNet50(num_classes, hidden_features)

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(device)

# Define weights for weighted cross entropy loss
# {'Figurines': 12847, 'Heads': 7713, 'Human Parts': 898, 'Jewelry': 7279, 'Reliefs': 2111, 
# 'Seal Stones - Seals - Stamps': 3968, 'Statues': 1473, 'Tools': 724, 
# 'Vases': 18926, 'Weapons': 1119}
# TotalImages = 57058 
# {0: 'Accessories', 1: 'Inscriptions', 2: 'Tools'}
# {'Accessories': 787, 'Inscriptions': 787, 'Tools': 878}
TotalImages = 57058 
Fw = TotalImages/12847
Hw = TotalImages/7713
HPw = TotalImages/898
Jw = TotalImages/7279
Rw = TotalImages/2111
SSw= TotalImages/3968
Sw = TotalImages/1473
Tw = TotalImages/724
Vw = TotalImages/18926
Ww =TotalImages/1119

weights = torch.tensor([Fw, Hw, HPw, Jw, Rw, SSw, Sw, Tw, Vw, Ww])
weights = weights.to(device)

# TotalAccessoriesWeight = 2452/787
# TotalInscriptionsWeight = 2452/787
# TotalToolsWeight = 2452/878
# TotalWeight = 787 + 787 + 878
# weights = torch.tensor([TotalAccessoriesWeight, TotalInscriptionsWeight, TotalToolsWeight])
# weights = weights.to(device)


# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.SGD(model.custom_classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Generate a timestamp to include in the log file name
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = "train_val_logs"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")

#--------------------------------------------------------------------------------------------------------------------------
# Writing results to log file
#--------------------------------------------------------------------------------------------------------------------------
best_val_accuracy = 0.0  # Initialise best validation accuracy

with open(log_file, 'a') as log:
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
        scheduler.step()  # Step the learning rate scheduler
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')
    
     # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/best_model.pth')
            log.write(f'\nSaved Best Model with Validation Accuracy: {val_accuracy:.2f}%\n')

    # Extract all info from test function     
    test_accuracy, confusion_matrix_df, class_names, misclassified_images, misclassified_labels, misclassified_predictions, misclassified_paths, top_n_info = test(model, test_loader, device, top_n=3)

    # Write confusion matrix and test accuracy to log file 
    log.write(f'Test Accuracy: {test_accuracy:.2f}%\n')

    log.write("\nVariables Used:\n")
    log.write(f'Number of Epochs: {args.num_epochs}\n')
    log.write(f'Number of Classes: {args.num_classes}\n')
    log.write(f'Hidden Features: {args.hidden_features}\n')
    log.write(f'Learning Rate: {args.lr}\n')

    # Write confusion matrix to log file
    log.write('\nConfusion Matrix:\n')
    log.write(confusion_matrix_df.to_string())
    log.write('\n')

    # Extract the top N predictions and probabilities from the dictionary
    all_top_n_predictions = top_n_info['all_top_n_predictions']
    all_top_n_probabilities = top_n_info['all_top_n_probabilities']
    correctly_classified_indices = top_n_info['correctly_classified_indices']
    incorrectly_classified_indices = top_n_info['incorrectly_classified_indices']
    paths = top_n_info['paths']
    confidence_correct = top_n_info['confidence_correct']
    confidence_incorrect = top_n_info['confidence_incorrect']
    borderline_cases = top_n_info['borderline_cases']
    all_labels = top_n_info['all_labels']
    cm = top_n_info['confusion_matrix']

    # Get the top N predictions and probabilities for correctly classified images
    correct_top_n_predictions, correct_top_n_probabilities = get_top_n_subset(correctly_classified_indices, all_top_n_predictions, all_top_n_probabilities)

    # Get the top N predictions and probabilities for incorrectly classified images
    incorrect_top_n_predictions, incorrect_top_n_probabilities = get_top_n_subset(incorrectly_classified_indices, all_top_n_predictions, all_top_n_probabilities)

     # Write correctly classified images info to log file
    top_n = 3
    log.write("\nCorrectly Classified Images\n")
    for i, idx in enumerate(correctly_classified_indices[:5]):  # Limiting to first 5 for brevity
        log.write(f"Image index: {idx}\n")
        log.write(f"Top {top_n} Predictions: {correct_top_n_predictions[i]}\n")
        log.write(f"Top {top_n} Probabilities: {correct_top_n_probabilities[i]}\n\n")

    # Write incorrectly classified images info to log file
    log.write("\nIncorrectly Classified Images\n")
    for i, idx in enumerate(incorrectly_classified_indices[:5]):  # Limiting to first 5 for brevity
        log.write(f"Image index: {idx}\n")
        log.write(f"Top {top_n} Predictions: {incorrect_top_n_predictions[i]}\n")
        log.write(f"Top {top_n} Probabilities: {incorrect_top_n_probabilities[i]}\n\n")
    
    # Write top 5 most confident correct predictions to log file
    log.write("\nTop 5 Most Confident Correct Predictions\n")
    for confidence, idx in confidence_correct[:5]:
        log.write(f"Confidence: {confidence:.4f}, Image index: {idx}, Path: {paths[idx]}\n")
        log.write(f"Top {top_n} Predictions: {all_top_n_predictions[idx]}\n")
        log.write(f"Top {top_n} Probabilities: {all_top_n_probabilities[idx]}\n\n")

    # Write bottom 5 least confident incorrect predictions to log file
    log.write("\nBottom 5 Least Confident Incorrect Predictions\n")
    for confidence, idx in confidence_incorrect[:5]:
        log.write(f"Confidence: {confidence:.4f}, Image index: {idx}, Path: {paths[idx]}\n")
        log.write(f"Top {top_n} Predictions: {all_top_n_predictions[idx]}\n")
        log.write(f"Top {top_n} Probabilities: {all_top_n_probabilities[idx]}\n\n")

    # Filter and write top 5 borderline correct cases to log file
    log.write("\nTop 5 Borderline Correct Cases\n")
    borderline_correct_cases = [case for case in borderline_cases if case[2]]  # Filter for correct predictions
    for confidence_diff, idx, is_correct in borderline_correct_cases[:5]:
        log.write(f"Confidence Difference: {confidence_diff:.4f}, Image index: {idx}, Path: {paths[idx]}, Correct: {is_correct}\n")
        log.write(f"Top {top_n} Predictions: {all_top_n_predictions[idx]}\n")
        log.write(f"Top {top_n} Probabilities: {all_top_n_probabilities[idx]}\n\n")

    # Filter and write top 5 borderline incorrect cases to log file
    log.write("\nTop 5 Borderline Incorrect Cases\n")
    borderline_incorrect_cases = [case for case in borderline_cases if not case[2]]  # Filter for incorrect predictions
    for confidence_diff, idx, is_correct in borderline_incorrect_cases[:5]:
        log.write(f"Confidence Difference: {confidence_diff:.4f}, Image index: {idx}, Path: {paths[idx]}, Correct: {is_correct}\n")
        log.write(f"Top {top_n} Predictions: {all_top_n_predictions[idx]}\n")
        log.write(f"Top {top_n} Probabilities: {all_top_n_probabilities[idx]}\n\n")

#--------------------------------------------------------------------------------------------------------------------------
# Visualising the distribution of correct labels 
#--------------------------------------------------------------------------------------------------------------------------  
    # Extract correct class labels for visualization
    correct_labels = [all_labels[idx] for idx in correctly_classified_indices]

    # Count frequency of each class in correctly classified labels
    class_counts = Counter(correct_labels)

    # Plot the distribution of correctly classified classes
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correctly Classified Classes')
    plt.xticks(rotation=45)
    plt.savefig('plots/DoC_130624.png')

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('plots/cm_130624.png')


#--------------------------------------------------------------------------------------------------------------------------
# Plot the saliency maps
#--------------------------------------------------------------------------------------------------------------------------  
# Visualize Grad-CAM for specific cases
#visualise_gradcam(model, test_loader, top_n_info, device, num_images=5)

#visualise_misclassified(misclassified_paths, misclassified_labels, misclassified_predictions, max_images=5)

