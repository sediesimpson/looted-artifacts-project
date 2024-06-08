import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from modelcomplete import *
from sklearn.metrics import confusion_matrix
import cv2

class GradCamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        self.pretrained = model.resnet50  # Access the ResNet50 part of the CustomResNet50 model
        self.layerhook.append(self.pretrained.layer4[2].register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out


def generate_gradcam(gcmodel, image, target_class):
    gcmodel.zero_grad()
    out, acts = gcmodel(image)
    loss = F.cross_entropy(out, target_class)
    loss.backward()

    acts = acts.detach().cpu()
    grads = gcmodel.get_act_grads().detach().cpu()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))

    return heatmap

def visualise_cam_on_image(image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_image = heatmap * alpha + np.float32(image) * (1 - alpha)
    overlayed_image = overlayed_image / np.max(overlayed_image)
    return np.uint8(255 * overlayed_image)


def visualise_gradcam(model, test_loader, top_n_info, device, num_images=5):
    gcmodel = GradCamModel(model).to(device)
    
    indices_to_visualise = []

    indices_to_visualise.extend([idx for _, idx in top_n_info['confidence_correct'][:num_images]])
    indices_to_visualise.extend([idx for _, idx in top_n_info['confidence_incorrect'][:num_images]])
    indices_to_visualise.extend([idx for _, idx, _ in top_n_info['borderline_cases'][:num_images]])

    indices_to_visualise = list(set(indices_to_visualise))  # Remove duplicates

    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        for idx in indices_to_visualise:
            if idx < len(images):
                input_image = images[idx].unsqueeze(0)
                target_class = labels[idx]
                cam = generate_gradcam(gcmodel, input_image, torch.tensor([target_class], dtype=torch.long, device=device))

                # Convert image to numpy array for visualization
                input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
                input_image_np = input_image_np - np.min(input_image_np)
                input_image_np = input_image_np / np.max(input_image_np)

                # Visualize the CAM on the image
                overlayed_image = visualise_cam_on_image(input_image_np, cam)

                # Display the image
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(input_image_np)
                plt.title(f'Original Image - True Label: {labels[idx].item()}')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(overlayed_image)
                plt.title(f'Grad-CAM - True Label: {labels[idx].item()}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('plots/gradcam2.png')
                plt.close()
