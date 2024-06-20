import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from collections import defaultdict
import cv2

class CustomImageDataset2(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(cls for cls in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cls)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths, self.labels = self.get_image_paths_and_labels()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.num_classes = len(self.classes)

    def get_image_paths_and_labels(self):
        label_dict = defaultdict(lambda: np.zeros(len(self.classes), dtype=int))
        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, cls_name)
            for root, _, files in os.walk(cls_path):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, self.root_dir)
                        label_dict[relative_path][self.class_to_idx[cls_name]] = 1  # Set the label for the class
        img_paths = list(label_dict.keys())
        #labels = list(label_dict.values())
        labels = [tuple(label) for label in label_dict.values()]  # Convert numpy arrays to tuples
        return img_paths, labels
    
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
            # Uncomment the line below to apply background subtraction
            # image = self.background_subtraction(img_path)
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
        # Method to print the labels and their corresponding indices
        label_info = {self.class_to_idx[cls_name]: cls_name for cls_name in self.classes}
        return label_info

    def count_images_per_label(self):
        # Method to count the number of images for each label
        label_counts = {cls_name: 0 for cls_name in self.classes}
        for labels in self.labels:
            for idx, label in enumerate(labels):
                if label == 1:
                    label_counts[self.classes[idx]] += 1
        return label_counts
    

# if __name__ == '__main__':

#     rootdir = '/rds/user/sms227/hpc-work/dissertation/data/TD10A'
#     dataset = CustomImageDataset(rootdir)
#     img_paths, labels = dataset.get_image_paths_and_labels() 
#     print(dataset.count_images_per_label())
#     #print(dataset.classes)




# from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler   

# batch_size = 32 
# root_dir = "/rds/user/sms227/hpc-work/dissertation/data/TD10A"
# dataset = CustomImageDataset(root_dir)
# test_split = 0.5
# validation_split = 0.4
# shuffle_dataset = True
# random_seed = 42

# # Create data indices for training, validation, and test splits
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# test_split_idx = int(np.floor(test_split * dataset_size))
# validation_split_idx = int(np.floor(validation_split * (dataset_size - test_split_idx)))

# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)

# test_indices = indices[:test_split_idx]
# train_val_indices = indices[test_split_idx:]
# train_indices = train_val_indices[validation_split_idx:]
# val_indices = train_val_indices[:validation_split_idx]

# # Create data samplers and loaders
# train_sampler = SubsetRandomSampler(train_indices)

# train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

# # def count_labels_in_loader(loader, class_to_idx):
# #     # Initialize label_counts using the class indices directly
# #     label_counts = {idx: 0 for idx in class_to_idx.values()}
# #     for _, labels, _ in loader:
# #         for label in labels.numpy():
# #             if label in label_counts:
# #                 label_counts[label] += 1
# #     return label_counts

# def count_labels_in_loader(loader, class_to_idx):
#     # Initialize label_counts using the class indices directly
#     label_counts = {idx: 0 for idx in class_to_idx.values()}
#     for _, labels, _ in loader:
#         for label in labels:
#             if label in label_counts:
#                 label_counts[label] += 1
#     return label_counts


# # Check label distribution in each loader
# train_label_counts = count_labels_in_loader(train_loader, dataset.class_to_idx)
# print("Training label distribution:", train_label_counts)

# specific_path = "/rds/user/sms227/hpc-work/dissertation/data/TD10A/Figurines/Ειδώλια Κλασσικά - Ετρουσκικά/Ειδώλια Γυναικεία Λίθινα/CAHN, BASEL, 05.11.2011, 212.2.png"
# labels = dataset.get_label_for_path(specific_path)
# print(f"Labels for {specific_path}: {labels}")
