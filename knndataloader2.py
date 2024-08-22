import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import numpy as np

class CustomImageDatasetTrain(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels = self.get_image_paths_and_labels()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.classes = list(self.label_to_idx.keys())
      

    def get_image_paths_and_labels(self):
        img_paths = []
        labels = []
        files = [name for name in glob.glob(DATASET)]
        files.sort()

        train_size = round(len(files) * 0.8)
        rng = np.random.default_rng(seed=42)
        trainset = rng.choice(files, size=train_size, replace=False, shuffle=False)

        for file in trainset:
            for root, _, files in os.walk(str(file)):
                for finalfile in files:
                    if finalfile.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and finalfile != '.DS_Store':
                            full_path = os.path.join(root, finalfile)
                            img_paths.append(full_path)
                            label = os.path.basename(root)
                            labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        return image, label_idx, img_path

    def count_images_per_label(self):
            # Method to count the number of images for each label
            label_counts = {cls_name: 0 for cls_name in self.classes}
            for label in self.labels:
                label_counts[label] += 1
            return label_counts
    
    def count_unique_labels(self):
      # Method to count the number of unique labels
      unique_labels = set(self.labels)
      return len(unique_labels)


class CustomImageDatasetTest(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels = self.get_image_paths_and_labels()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.classes = list(self.label_to_idx.keys())
      

    def get_image_paths_and_labels(self):
        img_paths = []
        labels = []
        files = [name for name in glob.glob(DATASET)]
        files.sort()

        train_size = round(len(files) * 0.8)
        rng = np.random.default_rng(seed=42)
        trainset = rng.choice(files, size=train_size, replace=False, shuffle=False)
        #print(trainset)
        #print(); print()
        testset = np.array([f for f in files if f not in trainset])
        #print(testset)


        for file in testset:
            for root, _, files in os.walk(str(file)):
                for finalfile in files:
                    if finalfile.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and finalfile != '.DS_Store':
                            full_path = os.path.join(root, finalfile)
                            img_paths.append(full_path)
                            label = os.path.basename(root)
                            labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        return image, label_idx, img_path

    def count_images_per_label(self):
            # Method to count the number of images for each label
            label_counts = {cls_name: 0 for cls_name in self.classes}
            for label in self.labels:
                label_counts[label] += 1
            return label_counts
    
    def count_unique_labels(self):
      # Method to count the number of unique labels
      unique_labels = set(self.labels)
      return len(unique_labels)

DATASET = "/rds/user/sms227/hpc-work/dissertation/data/duplicatedata/*"
# dataset = CustomImageDatasetTest(root_dir="/rds/user/sms227/hpc-work/dissertation/data/duplicatedata")
# dataloader = DataLoader(dataset, batch_size=32)

# # Count the number of images per label
# label_counts = dataset.count_images_per_label()
# print("Number of images per label:", label_counts)

# # Count the number of unique labels
# num_unique_labels = dataset.count_unique_labels()
# print("Number of unique labels in the dataset:", num_unique_labels)