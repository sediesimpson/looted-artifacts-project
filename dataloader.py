from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = self.get_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_image_paths(self):
        img_paths = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, cls_name)
            for root, _, files in os.walk(cls_path):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                        img_paths.append((os.path.join(root, file), cls_name))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, cls_name = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[cls_name]
        return self.transform(image), label
