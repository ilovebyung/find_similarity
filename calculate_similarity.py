import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_image_similarity(image_path1, image_path2):
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Check if images are loaded successfully
    if img1 is None or img2 is None:
        print("Error: Unable to load one or both images.")
        return None

    # Resize images to have the same dimensions
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    score, _ = ssim(gray1, gray2, full=True)

    return score

# Example usage
import os
files = os.listdir()
images = []
for file in files:
  if file.endswith('jpg'):
    images.append(file)

image1_path = images[0]
image2_path = images[2]

similarity_score = calculate_image_similarity(image1_path, image2_path)

if similarity_score is not None:
    print(f"Similarity score: {similarity_score:.4f}")
    print(f"Percentage similarity: {similarity_score * 100:.2f}%")



