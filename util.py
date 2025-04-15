import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def match_and_align_image(image, template):
    # Template matching
    h, w = template.shape[::]
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    matched_area = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # Image alignment
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(matched_area, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:20]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(matched_area, M, (template.shape[1], template.shape[0]))
    
    return aligned_image

if __name__ == '__main__':
    
    template = cv2.imread('template.jpg', 0)
    image = cv2.imread('sample.jpg', 0)
    
    plt.imshow(template, cmap='gray')
    plt.imshow(image, cmap='gray')
    
    result = match_and_align_image(image, template)
    plt.imshow(result, cmap='gray')