import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def background_subtraction(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None, None

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the red background color
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define the range for the black background color
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Define the range for the white background color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])

    # Create masks for the red, black, and white backgrounds
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine all masks
    combined_mask = mask_red | mask_black | mask_white

    # Invert the combined mask to get the foreground
    mask_inv = cv2.bitwise_not(combined_mask)

    # Use the mask to extract the foreground
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)

    # Convert the foreground to PIL format
    foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))


    # Convert the original image and foreground to PIL format
    original_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))

    return original_pil, foreground_pil


def visualize_original_and_foreground(image_path):
    original_image, foreground_image = background_subtraction(image_path)
    if original_image and foreground_image:
        plt.figure(figsize=(10, 5))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        # Plot the foreground image
        plt.subplot(1, 2, 2)
        plt.imshow(foreground_image)
        plt.title("Foreground Image")
        plt.axis('off')

        plt.show()

# Example usage
query_image_path = "/Users/sedisimpson/Desktop/Dissertation Data/Test Dataset 5/Accessories/Barakat Volume-11, FZ210.JPG"
visualize_original_and_foreground(query_image_path)
