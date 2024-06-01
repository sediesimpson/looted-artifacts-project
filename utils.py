import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

#-----------------------------------------------------------------------------------------------------
# Grad-CAM Implementation
#-----------------------------------------------------------------------------------------------------
class Hook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        module.register_forward_hook(self.forward_hook)
        module.register_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output
        print(f"Forward Hook: Captured activations of shape {self.activations.shape}")

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        print(f"Backward Hook: Captured gradients of shape {self.gradients.shape}")

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor, image

# Update the get_gradcam_heatmap function to use the hook
def get_gradcam_heatmap(model, image_tensor, hook, target_class=None):
    model.eval()
    
    # Forward pass
    print("Performing forward pass...")
    outputs = model(image_tensor)
    
    # Get the index of the target class
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    
    # Backward pass to get gradients
    model.zero_grad()
    print("Performing backward pass...")
    target = outputs[0][target_class]
    target.backward(retain_graph=True)

    # Get the gradients and the activations from the hook
    gradients = hook.gradients
    activations = hook.activations

    # Debug print statements
    if gradients is None:
        print("Gradients are not captured.")
    else:
        print(f"Gradients shape: {gradients.shape}")

    if activations is None:
        print("Activations are not captured.")
    else:
        print(f"Activations shape: {activations.shape}")

    # Check if gradients and activations are not None
    if gradients is None or activations is None:
        raise ValueError("Gradients or activations are not captured properly.")

    # Global Average Pooling on the gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the activations to get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap

def apply_heatmap_on_image(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = np.array(image) * (1 - alpha) + heatmap * alpha
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img

def visualise_gradcam(model, image_path, device, target_layer, class_names, target_class=None):
    image_tensor, image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    target_layer.gradients = None

    # Forward and backward pass
    heatmap = get_gradcam_heatmap(model, image_tensor, target_layer, target_class)

    # Apply heatmap on the image
    superimposed_img = apply_heatmap_on_image(image, heatmap)

    # Display the result
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM for {image_path}")
    plt.axis('off')
    plt.savefig('plots/gradcam.png')
