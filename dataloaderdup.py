import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import numpy as np
import cv2
from pprint import pprint

class CustomImageDatasetTrain(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels, self.label_counts = self.get_image_paths_and_labels()
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
        label_counts = []
        files = [name for name in glob.glob(DATASET)]
        files.sort()

        train_size = round(len(files) * 0.8)
        rng = np.random.default_rng(seed=42)
        trainset = rng.choice(files, size=train_size, replace=False, shuffle=False)

        for file in trainset:
            for root, _, files in os.walk(str(file)):
                finalfilecount = 0
                for finalfile in files:
                    if finalfile.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and finalfile != '.DS_Store':
                        full_path = os.path.join(root, finalfile)
                        img_paths.append(full_path)

                        label = os.path.basename(root)
                        labels.append(label)

                        finalfilecount +=1

                for _ in range(finalfilecount):
                    label_counts.append(finalfilecount)

        return img_paths, labels, label_counts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        label_counts = self.label_counts[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        return image, label, label_idx, img_path, label_counts
    
    def count_unique_labels(self):
      # Method to count the number of unique labels
      unique_labels = set(self.labels)
      return len(unique_labels)


class CustomImageDatasetTest(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels, self.label_counts = self.get_image_paths_and_labels()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.classes = list(self.label_to_idx.keys())
    
    def background_subtraction(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        combined_mask = mask_red | mask_black | mask_white
        mask_inv = cv2.bitwise_not(combined_mask)
        foreground = cv2.bitwise_and(image, image, mask=mask_inv)
        foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
        return foreground_pil

    def get_image_paths_and_labels(self):
        img_paths = []
        labels = []
        label_counts = []
        files = [name for name in glob.glob(DATASET)]
        files.sort()

        train_size = round(len(files) * 0.8)
        rng = np.random.default_rng(seed=42)
        trainset = rng.choice(files, size=train_size, replace=False, shuffle=False)
        testset = np.array([f for f in files if f not in trainset])

        for folder in testset:
            for root, _, files in os.walk(str(folder)):
                file_count = 0

                for f in files:
                    if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and f != '.DS_Store':
                        img_path = os.path.join(root, f)
                        img_paths.append(img_path)

                        label = os.path.basename(root)
                        labels.append(label)
                        file_count +=1

                for _ in range(file_count):
                    label_counts.append(file_count)
        
        # for i, l, c in zip(img_paths, labels, label_counts):
        #     print(f"image: {i}, label: {l}, counts: {c}")

        return img_paths, labels, label_counts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        label_counts = self.label_counts[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = cv2.imread(img_path)
            # if image is None:
            #     print(f"Error loading image {image_path}")
            #     return None
            image = self.background_subtraction(image)
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        return image, label, label_idx, img_path, label_counts

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
