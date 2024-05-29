from finaldataloader import CustomImageDataset
import torch
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

