import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_image_similarity(template_path, new_image_path):
    # Read the images
    img1 = cv2.imread(template_path)
    img2 = cv2.imread(new_image_path)

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

image_1 = images[0]
image_2 = images[2]

similarity_score = calculate_image_similarity(image_1, image_2)

if similarity_score is not None:
    print(f"Similarity score: {similarity_score:.4f}")
    print(f"Percentage similarity: {similarity_score * 100:.2f}%")


######### cv2 implementation ######### 
import cv2

def compare_images(template_path, new_image_path):
    """Compares a new image against a template and returns a similarity score.

    Args:
    template_path: Path to the template image.
    new_image_path: Path to the new image to be compared.

    Returns:
    A similarity score between 0 and 1.
    """

    template = cv2.imread(template_path)
    new_image = cv2.imread(new_image_path)

    # Convert images to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(new_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Calculate similarity score (higher is better)
    similarity_score = max_val

    return similarity_score

# Example usage
template_path = "template.jpg"
new_image_path = "new_image.jpg"

score = compare_images(template_path, new_image_path)
print("Similarity score:", score)