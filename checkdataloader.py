from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
from dataloader import CustomImageDataset

# Define your dataset
root_dir =  '/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 2'
custom_dataset = CustomImageDataset(root_dir)

# Define DataLoader
batch_size = 32
train_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the DataLoader
for batch_idx, (images, labels) in enumerate(train_loader):
    # Print or inspect a few batches of data
    print("Batch Index:", batch_idx)
    print("Batch Images Shape:", images.shape)
    print("Batch Labels Shape:", labels.shape)

    # Optionally, you can visualize or inspect some sample images
    # Assuming images are in tensor format, you can use matplotlib for visualization
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert images from tensor to numpy array and transpose if needed
    np_images = images.numpy() if isinstance(images, torch.Tensor) else images
    np_images = np.transpose(np_images, (0, 2, 3, 1)) if len(np_images.shape) == 4 else np_images

    # Visualize some sample images from the batch
    num_samples = min(len(images), 4)  # Visualize up to 4 images
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 4))
    for i in range(num_samples):
        axes[i].imshow(np_images[i])
        axes[i].set_title("Label: {}".format(labels[i]))
        axes[i].axis('off')
    plt.show()
