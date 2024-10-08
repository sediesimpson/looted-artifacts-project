import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm


# # Define a custom classifier module
# class CustomClassifier(nn.Module):
#     def __init__(self, input_features, hidden_features, num_classes):
#         super(CustomClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_features, hidden_features)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc2 = nn.Linear(hidden_features, num_classes)
#         #self.sigmoid = nn.Sigmoid()  # For multi-label classification

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         #x = self.sigmoid(x)  # Apply sigmoid activation
#         return x

# # Define a model wrapper that combines ResNet and CustomClassifier
# class CustomResNet50(nn.Module):
#     def __init__(self, num_classes, hidden_features):
#         super(CustomResNet50, self).__init__()
#         self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         # Freeze the convolutional layers
#         for param in self.resnet50.parameters():
#             param.requires_grad = False

#         for param in self.resnet50.fc.parameters():
#             param.requires_grad = True


#         num_features = self.resnet50.fc.in_features  # This should be 2048 for ResNet50
#         self.resnet50.fc = nn.Linear(num_features, num_classes)
#         #self.resnet50.fc = nn.Identity()  # Remove the existing fully connected layer
#         #self.custom_classifier = CustomClassifier(num_features, hidden_features, num_classes)

#     def forward(self, x):
#         # Extract features from the second-to-last layer
#         x = self.resnet50(x)
#         #x = self.resnet50.avgpool(self.resnet50.layer4(self.resnet50.layer3(self.resnet50.layer2(self.resnet50.layer1(self.resnet50.maxpool(self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x)))))))))
#         #x = torch.flatten(x, 1)
#         #x = self.custom_classifier(x)
#         return x

# def get_resnet(num_classes):
#     weights = ResNet50_Weights.DEFAULT
#     model = models.resnet50(weights=weights)
#     # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# # Freeze the convolutional layers
#     for param in model.parameters():
#         param.requires_grad = False

#     for param in model.fc.parameters():
#         param.requires_grad = True

#     num_features = model.fc.in_features  # This should be 2048 for ResNet50
#     model.fc = nn.Linear(num_features, num_classes)

#     return model 


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