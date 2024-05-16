import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

class CustomClassifierEmbedding(nn.Module):
    def __init__(self, input_features, hidden_features):
        super(CustomClassifierEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x  # Returns the embedding vector

class ResNet50Embedding(nn.Module):
    def __init__(self, hidden_features):
        super(ResNet50Embedding, self).__init__()
        # Loading the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Removing the final fully connected layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        # Freezing the convolutional layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Assume output features from ResNet50 are 2048 after avgpool
        self.custom_classifier = CustomClassifierEmbedding(2048, hidden_features)

    def forward(self, x):
        # Extracting features from the modified ResNet50 base
        x = self.resnet50(x)
        x = torch.flatten(x, 1)
        x = self.custom_classifier(x)
        return x
