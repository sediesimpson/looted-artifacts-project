import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class CustomImageDataset3(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths, self.labels, self.class_to_idx = self.get_image_paths_and_labels()
        self.classes = list(self.class_to_idx.keys())
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.num_classes = len(self.classes)
        self.remove_empty_labels()

    def get_image_paths_and_labels(self):
        img_paths = []
        labels = []
        class_to_idx = {}
        
        for root, _, files in os.walk(self.root_dir):
            class_name = os.path.basename(root)
            if class_name == os.path.basename(self.root_dir):
                continue
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            
            label = np.zeros(len(class_to_idx), dtype=int)
            label[class_to_idx[class_name]] = 1
            
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                    full_path = os.path.join(root, file)
                    img_paths.append(os.path.relpath(full_path, self.root_dir))
                    labels.append(tuple(label))
                    
        return img_paths, labels, class_to_idx
    
    def background_subtraction(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            return None
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
   
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        label = torch.FloatTensor(self.labels[idx])
        return image, label, img_path

    def get_label_for_path(self, img_path):
        labels = set()
        for i, path in enumerate(self.img_paths):
            if os.path.basename(path) == os.path.basename(img_path):
                label_vector = self.labels[i]
                label_indices = np.where(label_vector == 1)[0]
                labels.update(self.classes[i] for i in label_indices)
        return list(labels)
    
    def get_label_info(self):
        label_info = {self.class_to_idx[cls_name]: cls_name for cls_name in self.classes}
        return label_info
    
    def get_label_names(self):
        return list(self.classes)

    def count_images_per_label(self):
        label_counts = {cls_name: 0 for cls_name in self.classes}
        for labels in self.labels:
            for idx, label in enumerate(labels):
                if label == 1:
                    label_counts[self.classes[idx]] += 1
        return label_counts

    def remove_empty_labels(self):
        # Count the number of images per label
        label_counts = self.count_images_per_label()
        
        # Identify labels with zero images
        labels_to_keep = {cls_name for cls_name, count in label_counts.items() if count > 0}

        # Create a mapping from old indices to new indices
        old_class_to_idx = self.class_to_idx.copy()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(labels_to_keep))}
        self.classes = list(self.class_to_idx.keys())

        # Update labels to match new class_to_idx
        new_labels = []
        for label in self.labels:
            new_label = np.zeros(len(self.class_to_idx), dtype=int)
            for cls_name, old_idx in old_class_to_idx.items():
                if old_idx < len(label) and label[old_idx] == 1:
                    if cls_name in self.class_to_idx:
                        new_label[self.class_to_idx[cls_name]] = 1
            new_labels.append(tuple(new_label))

        # Update img_paths and labels to only include the kept labels
        self.img_paths = [img_path for img_path, label in zip(self.img_paths, new_labels) if any(label)]
        self.labels = [label for label in new_labels if any(label)]
        self.num_classes = len(self.classes)

# # Replace this with the actual path to your dataset
# root_dir = "/rds/user/sms227/hpc-work/dissertation/data/la_data"
# dataset = CustomImageDataset3(root_dir)

# # Function to print dataset information
# def print_dataset_info(dataset):
#     print("Number of classes:", dataset.num_classes)
#     print("Class to index mapping:", dataset.class_to_idx)
#     print("Number of images:", len(dataset))
#     print("Images per class:", dataset.count_images_per_label())

# # Print initial dataset information
# print("Initial dataset information:")
# print_dataset_info(dataset)

# # Remove empty labels
# dataset.remove_empty_labels()

# # Print updated dataset information
# print("\nUpdated dataset information:")
# print_dataset_info(dataset)

