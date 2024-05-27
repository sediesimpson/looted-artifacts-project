from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
import numpy as np
import cv2

# TODO

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(cls for cls in os.listdir(root_dir) if cls != '.DS_Store')
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = self.get_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_image_paths(self):
        img_paths = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, cls_name)
            for root, _, files in os.walk(cls_path):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                        img_paths.append((os.path.join(root, file), cls_name))
        return img_paths

    def background_subtraction(self, image_path):
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                return None

            # Convert the image to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the range for the red background color
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            # Define the range for the black background color
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])

            # Define the range for the white background color
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 20, 255])

            # Create masks for the red, black, and white backgrounds
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = mask_red1 | mask_red2

            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            # Combine all masks
            combined_mask = mask_red | mask_black | mask_white

            # Invert the combined mask to get the foreground
            mask_inv = cv2.bitwise_not(combined_mask)

            # Use the mask to extract the foreground
            foreground = cv2.bitwise_and(image, image, mask=mask_inv)

            # Convert the foreground to PIL format
            foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))

            return foreground_pil


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, cls_name = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        label = self.class_to_idx[cls_name]
        return self.transform(image), label, img_path

    def get_label_info(self):
        # Method to print the labels and their corresponding indices
        label_info = {self.class_to_idx[cls_name]: cls_name for cls_name in self.classes}
        return label_info

    def count_images_per_label(self):
        # Method to count the number of images for each label
        label_counts = {cls_name: 0 for cls_name in self.classes}
        for _, cls_name in self.img_paths:
            label_counts[cls_name] += 1
        return label_counts
