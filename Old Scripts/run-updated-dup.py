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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
from multidataloader import *
import argparse
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import sys
from modelcompletedup import CustomResNetClassifier
from tqdm import tqdm
#--------------------------------------------------------------------------------------------------------------------------
# Define argsparser
#--------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description = 'Running Baseline Models')

parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--root_dir', type=str, default="/rds/user/sms227/hpc-work/dissertation/data/la_data")
parser.add_argument('--validation_split', type=int, default=0.1)
parser.add_argument('--test_split', type=int, default=0.1)
parser.add_argument('--shuffle_dataset', type=bool, default=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=28)
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
root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"
dataset = CustomImageDataset2(root_dir)

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
#--------------------------------------------------------------------------------------------------------------------------
# Function to save a checkpoint
#--------------------------------------------------------------------------------------------------------------------------
def save_checkpoint(epoch, model, optimizer, loss, best_model, folder="checkpoints", filename="model.pt"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

    # If this is the best model, save a separate checkpoint
    if best_model:
        best_path = os.path.join(folder, "best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, best_path)
#--------------------------------------------------------------------------------------------------------------------------
# Training, Validation and Testing Functions
#--------------------------------------------------------------------------------------------------------------------------
#class_names = ['Accessories', 'Inscriptions', 'Tools']
#class_names = ['Figurines', 'Heads', 'Human Parts', 'Jewelry', 'Reliefs', 'Seal Stones - Seals - Stamps', 'Statues', 'Tools', 'Vases', 'Weapons']
class_names = ['Accessories','Altars','Candelabra','Coins - Metals','Columns - Capitals',
'Decorative Tiles','Egyptian Coffins','Figurines','Fossils','Frescoes - Mosaics'
,'Heads','Human Parts','Inscriptions','Islamic','Jewelry','Manuscripts','Mirrors'
,'Musical Instruments','Oscilla','Other Objects','Reliefs','Sarcophagi - Urns',
'Sardinian Boats','Seal Stones - Seals - Stamps','Statues','Tools','Vases','Weapons']

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total = 0 
    print('Device in training loop:', device)

    for i, (images, labels, _) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device).float()  # float labels for BCEWithLogitsLoss
        # Forward pass
        outputs = model(images)
        # Compute the loss using BCEWithLogitsLoss 
        loss = criterion(outputs, labels)
        # Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  # Accumulate loss 


    average_loss = running_loss / len(train_loader)  # Compute the average loss for the epoch
    print('Training Loss: {:.4f}'.format(average_loss))
    return average_loss


def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_step = len(valid_loader)
    total = 0
    print('Device in validation loop:', device)
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device).float()  # Float labels for BCEWithLogitsLoss

            outputs = model(images)

            loss = criterion(outputs, labels) 

            running_loss += loss.item() 
            probs = torch.sigmoid(outputs) 
            predicted = probs.round()

            correct_predictions += (predicted == labels).sum().item()

            total += labels.size(1) * batch_size

    
    # Compute the average loss for the epoch
    val_loss = running_loss / len(valid_loader) 
    # Compute the accuracy
    val_accuracy = correct_predictions / total 
    
    print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_accuracy))
    
    return val_loss, val_accuracy

def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    
    print('Device in test loop:', device)
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            predicted = probs.round()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='samples')
    recall = recall_score(all_labels, all_predictions, average='samples')
    f1 = f1_score(all_labels, all_predictions, average='samples')
    
    # Calculate per-class accuracy
    per_class_accuracy = (all_predictions == all_labels).sum(axis=0) / all_labels.shape[0]

    # Generate normalized confusion matrix
    cm = confusion_matrix(all_labels.argmax(axis=1), all_predictions.argmax(axis=1), labels=range(28), normalize='true')

    # Print results
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(accuracy, precision, recall, f1))
    
    # Print per-class accuracy
    for i, acc in enumerate(per_class_accuracy):
        print(f'Accuracy for class {i}: {acc:.4f}')
    
    return accuracy, precision, recall, f1, per_class_accuracy, cm
#--------------------------------------------------------------------------------------------------------------------------
# Running the model
#--------------------------------------------------------------------------------------------------------------------------
num_classes = args.num_classes
hidden_features = args.hidden_features
learning_rate = args.lr
num_epochs = args.num_epochs 

print("\nVariables Used:\n")
print(f'Number of Epochs: {args.num_epochs}\n')
print(f'Number of Classes: {args.num_classes}\n')
print(f'Hidden Features: {args.hidden_features}\n')
print(f'Learning Rate: {args.lr}\n')

hidden_dim = 256  # Intermediate hidden layer size
weights = models.ResNet50_Weights.DEFAULT  # Pre-trained weights
model = CustomResNetClassifier(hidden_dim, num_classes, weights=weights)

# Move the model to the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
print(device)
#--------------------------------------------------------------------------------------------------------------------------
# Weighting function
#--------------------------------------------------------------------------------------------------------------------------
# Get image paths and labels
# img_paths, labels = dataset.get_image_paths_and_labels()

# # Convert list of labels to a numpy array for easier manipulation
# labels_array = np.array(labels)

# # Calculate pos_weight for each class
# positive_counts = np.sum(labels_array, axis=0)

# negative_counts = labels_array.shape[0] - positive_counts
# pos_weight = negative_counts / positive_counts

# # Convert pos_weight to a tensor
# pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
# print('pos_weight_tensor:', pos_weight_tensor)

# Use pos_weight in BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
criterion = criterion.to(device)
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


best_val_accuracy = 0.0 

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
    print('-' * 30)

    # Check if the current model is the best
    best_model = val_accuracy > best_val_accuracy
    if best_model:
        best_val_accuracy = val_accuracy

    save_checkpoint(epoch, model, optimizer, val_loss, best_model, folder="checkpoints", filename="model_epoch_{}.pt".format(epoch))


accuracy, precision, recall, f1, per_class_accuracy, cm = test(model, test_loader, device)
print('-' * 30)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(28), yticklabels=range(28))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('plots/cm_dup.png')

sys.exit()


#--------------------------------------------------------------------------------------------------------------------------
# End here for now whilst testing validation accuracy
#--------------------------------------------------------------------------------------------------------------------------


def test(model, test_loader, device, top_n=3):
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_top_n_predictions = []
    all_top_n_probabilities = []
    all_paths = []
    correctly_classified_indices = []
    incorrectly_classified_indices = []
    confidence_correct = []
    confidence_incorrect = []
    borderline_cases = []
    per_class_accuracy = []
    cms = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)

            # Calculate sigmoid probabilities
            probabilities = torch.sigmoid(outputs)

            # Get the top N predictions and their probabilities
            top_n_probabilities, top_n_predictions = torch.topk(probabilities, top_n, dim=1)

            # Append the top N predictions and probabilities for each image
            all_top_n_predictions.extend(top_n_predictions.cpu().numpy())
            all_top_n_probabilities.extend(top_n_probabilities.cpu().numpy())

            # Apply a threshold to get binary predictions
            predicted = (probabilities > 0.5).int()
            total += labels.numel()
            correct += (predicted == labels.int()).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_paths.extend(paths)

            # Collect correctly and misclassified images indices
            for j in range(len(labels)):
                confidence = probabilities[j].max().item()
                confidence_diff = top_n_probabilities[j][0] - top_n_probabilities[j][1] if top_n > 1 else 1.0
                if (predicted[j] == labels[j].int()).all():
                    correctly_classified_indices.append(len(all_labels) - len(labels) + j)
                    confidence_correct.append((confidence, len(all_labels) - len(labels) + j))
                else:
                    incorrectly_classified_indices.append(len(all_labels) - len(labels) + j)
                    confidence_incorrect.append((confidence, len(all_labels) - len(labels) + j))
                borderline_cases.append((confidence_diff, len(all_labels) - len(labels) + j, (predicted[j] == labels[j].int()).all()))

    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the test images: {:.2f}%'.format(accuracy))

    # Collect confusion matrix information for each label independently
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)
    num_classes = all_labels_np.shape[1]

    # Calculate confusion matrix and per-class accuracy
    per_class_accuracy = {}
    for i in range(num_classes):
        cm = confusion_matrix(all_labels_np[:, i], all_predictions_np[:, i])
        cms.append(cm)

        true_positives = cm[1, 1]
        true_negatives = cm[0, 0]
        total_samples = cm.sum()
        per_class_accuracy[class_names[i]] = (true_positives + true_negatives) / total_samples


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
        "confusion_matrix": cms,
        "per_class_accuracy": per_class_accuracy 
    }

    return accuracy, cms, top_n_info

#--------------------------------------------------------------------------------------------------------------------------
# Function to visualise top N predictions and probabilities
#--------------------------------------------------------------------------------------------------------------------------
def get_top_n_subset(indices, top_n_predictions, top_n_probabilities):
    subset_top_n_predictions = [top_n_predictions[i] for i in indices]
    subset_top_n_probabilities = [top_n_probabilities[i] for i in indices]
    return subset_top_n_predictions, subset_top_n_probabilities








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
    test_accuracy, confusion_matrix_df, top_n_info = test(model, test_loader, device, top_n=3)

    # Write confusion matrix and test accuracy to log file 
    log.write(f'Test Accuracy: {test_accuracy:.2f}%\n')

    log.write("\nVariables Used:\n")
    log.write(f'Number of Epochs: {args.num_epochs}\n')
    log.write(f'Number of Classes: {args.num_classes}\n')
    log.write(f'Hidden Features: {args.hidden_features}\n')
    log.write(f'Learning Rate: {args.lr}\n')

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
    cms = top_n_info['confusion_matrix']
    per_class_accuracy = top_n_info['per_class_accuracy']

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

def display_confusion_matrices(cms, class_names, log_file):
    with open(log_file, 'a') as log:
        # Convert confusion matrices to DataFrames and write to the log file
        for i, cm in enumerate(cms):
            class_name = class_names[i] if i < len(class_names) else str(i)
            cm_df = pd.DataFrame(cm, 
                                 index=[f'True {class_name} Negative', f'True {class_name} Positive'], 
                                 columns=[f'Pred {class_name} Negative', f'Pred {class_name} Positive'])
            log.write(f'Confusion Matrix for Label {class_name}:\n{cm_df}\n\n')

    print(f'Confusion matrices have been written to {log_file}')
# Display the confusion matrices
display_confusion_matrices(cms, class_names, log_file)

def log_per_class_accuracy(per_class_accuracy, log_file):
    with open(log_file, 'a') as log:
        log.write('Per-class accuracy:\n')
        for class_name, accuracy in per_class_accuracy.items():
            log.write(f'{class_name}: {accuracy:.2f}\n')

    print(f'Per-class accuracy has been written to {log_file}')

# Write the per-class accuracy to the log file
log_per_class_accuracy(top_n_info["per_class_accuracy"], log_file)