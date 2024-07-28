import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define a custom ResNet model for embeddings
class CustomResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim, weights=None):
        super(CustomResNetEmbedding, self).__init__()
        self.resnet = models.resnet50(weights=weights)
        
        # Replace the fully connected layer with an embedding layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, embedding_dim)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

