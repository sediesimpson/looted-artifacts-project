import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

# Define a custom classifier module
class CustomClassifier(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define a model wrapper that combines ResNet and CustomClassifier
class CustomResNet18(nn.Module):
    def __init__(self, num_classes, hidden_features):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze the convolutional layers
        for param in self.resnet18.parameters():
            param.requires_grad = False

        num_features = self.resnet18.fc.in_features  # This should be 512 for ResNet18
        self.resnet18.fc = nn.Identity()  # Remove the existing fully connected layer
        self.custom_classifier = CustomClassifier(num_features, hidden_features, num_classes)

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.resnet18.avgpool(self.resnet18.layer4(self.resnet18.layer3(self.resnet18.layer2(self.resnet18.layer1(self.resnet18.maxpool(self.resnet18.relu(self.resnet18.bn1(self.resnet18.conv1(x)))))))))
        x = torch.flatten(x, 1)
        x = self.custom_classifier(x)
        return x