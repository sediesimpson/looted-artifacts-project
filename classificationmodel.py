import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm

class CustomResNetClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, weights=None):
        super(CustomResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights=weights)
        self.resnet.requires_grad_(False)  # Freeze all layers

        # Replace the fully connected layer with a custom sequential layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes)  # Output layer with one neuron per class for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.resnet(x)
        return x