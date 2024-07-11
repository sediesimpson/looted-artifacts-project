import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomImageDataset3(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths = self.get_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_image_paths(self):
        img_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')) and file != '.DS_Store':
                    full_path = os.path.join(root, file)
                    img_paths.append(full_path)
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        image = self.transform(image)
        return image, img_path

# # Assuming images are stored in 'path_to_images'
# dataset = CustomImageDataset3(root_dir="/rds/user/sms227/hpc-work/dissertation/data/1NN")
# dataloader = DataLoader(dataset, batch_size=32)

# # Iterate through the dataset
# for images, paths in dataloader:
#     # images: Tensor of shape [batch_size, 3, 224, 224]
#     # paths: List of image file paths
#     print(images.shape)
#     print(paths)
