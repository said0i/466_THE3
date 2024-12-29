import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

# Helper function to display images
def display_images(titles, images):
    plt.figure(figsize=(15, 10))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(2, 3, i+1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Gray-scale Morphology
def grayscale_morphology(image, operation, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# KMeans using RGB Features
def kmeans_rgb(image, k=2):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    segmented = kmeans.labels_.reshape(image.shape[:2])
    return segmented

# KMeans using LBP Features
def kmeans_lbp(image, k=2, radius=1, n_points=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_flattened = lbp.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(lbp_flattened)
    segmented = kmeans.labels_.reshape(gray.shape)
    return segmented

# Main function to process and compare methods
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply grayscale morphology
    morph_open = grayscale_morphology(gray, 'opening', kernel_size=7)
    morph_close = grayscale_morphology(gray, 'closing', kernel_size=7)

    # Apply KMeans with RGB
    kmeans_rgb_result = kmeans_rgb(image, k=2)

    # Apply KMeans with LBP
    kmeans_lbp_result = kmeans_lbp(image, k=2, radius=3, n_points=24)

    # Display results
    titles = ['Original', 'Morph Opening', 'Morph Closing', 'KMeans RGB', 'KMeans LBP']
    images = [image, morph_open, morph_close, kmeans_rgb_result, kmeans_lbp_result]
    display_images(titles, images)

# Process all images
image_folder = 'THE3_Images/'
image_paths = [image_folder + str(i) + '.png' for i in range(1,7)]
for path in image_paths:
    process_image(path)

