# Student1 Name: Ahmed Said Gençkol Student1 ID: 2539377
# Student2 Name: Batuhan Teberoğu Student2 ID: 2581056

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

#######################################
####//////IMAGE READ-WRITE\\\\\\\\#####
#######################################
input_folder = 'THE3_Images/'
output_folder = 'THE3_Outputs/'

def read_image(filename):
    img = cv2.imread(input_folder + filename, cv2.IMREAD_UNCHANGED)
    return img
#Write each image into its own folder
def write_image(img, filename):
    cv2.imwrite(output_folder+filename[0]+'/'+filename, img)


#######################################
####//////BASIC FUNCTIONS\\\\\\\\######
#######################################
#Apply Kmeans clustering to an image
def kmeans_clustering(image, k=2):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    Z = image.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    
    return res2
# extended LBP
def ELBP(src, radius, neighbors):
    neighbors = max(min(neighbors, 31), 1)
    rows, cols = src.shape
    dst = np.zeros((rows - 2 * radius, cols - 2 * radius), dtype=np.uint32)

    for n in range(neighbors):
        x = radius * np.cos(2.0 * np.pi * n / neighbors)
        y = -radius * np.sin(2.0 * np.pi * n / neighbors)
        fx, fy = int(np.floor(x)), int(np.floor(y))
        cx, cy = int(np.ceil(x)), int(np.ceil(y))
        tx, ty = x - fx, y - fy
        w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                t = (
                    w1 * src[i + fy, j + fx]
                    + w2 * src[i + fy, j + cx]
                    + w3 * src[i + cy, j + fx]
                    + w4 * src[i + cy, j + cx]
                )
                if (t > src[i, j]) and (abs(t - src[i, j]) > np.finfo(float).eps):
                    dst[i - radius, j - radius] += (1 << n)
    
    return dst

#why Uniform: https://stackoverflow.com/questions/32105338/uniform-lbp-with-scikit-image-local-binary-pattern-function
# skimage LBP function which works faster
def ELBP_skimage(src, radius, neighbors):
    lbp = local_binary_pattern(src, neighbors, radius, method='uniform')
    return lbp

#Morphological operations
def erode(img, kernel):
    kernel = np.ones(kernel,np.uint8)
    r_img = cv2.erode(img,kernel,iterations = 1)
    return r_img

def dilate(img, kernel):
    kernel = np.ones(kernel,np.uint8)
    d_img = cv2.dilate(img,kernel,iterations = 1)
    return d_img

def morph(img, kernel, morph_type):
    kernel = np.ones(kernel,np.uint8)
    o_img = cv2.morphologyEx(img, morph_type, kernel)
    return o_img

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

#Get Specific Channel of the Image
def get_channel(image, channel):
    if channel == 'red':
        return image[:, :, 2]
    elif channel == 'green':
        return image[:, :, 1]
    elif channel == 'blue':
        return image[:, :, 0]
    elif channel == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Invalid channel name')


# Helper function to display images
def display_images(titles, images):
    plt.figure(figsize=(15, 10))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot()
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# KMeans using RGB Features
def kmeans_rgb(image, k=2):
    #pixels = image.reshape(-1, 3)
    #kmeans = KMeans(n_clusters=k, random_state=0).fit(image)
    kmeans = kmeans_clustering(image, k)
    #segmented = kmeans.labels_.reshape(image.shape[:2])
    return kmeans

# KMeans using LBP Features
def kmeans_lbp(image, k=2, radius=1, n_points=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = get_channel(image, 'red')
    lbp= ELBP_skimage(gray, radius, n_points)
    lbp_flattened = lbp.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=k, random_state=0).fit(lbp_flattened)
    kmeans = kmeans_clustering(lbp_flattened, k)
    segmented = kmeans.reshape(gray.shape)
    return segmented

# a general function to process and compare methods bluntly
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = get_channel(image, 'red')
    # Apply grayscale morphology
    morph_open = grayscale_morphology(gray, 'opening', kernel_size=7)
    morph_close = grayscale_morphology(gray, 'closing', kernel_size=7)

    # Apply KMeans with RGB
    kmeans_rgb_result = kmeans_rgb(image, k=2)

    # Apply KMeans with LBP
    kmeans_lbp_result = kmeans_lbp(image, k=2, radius=2, n_points=8)

    # Display results
    titles = ['Original', 'Grayscale', 'Red', 'Morph Opening', 'Morph Closing', 'KMeans RGB', 'KMeans LBP']
    images = [image, gray, red, morph_open, morph_close, kmeans_rgb_result, kmeans_lbp_result]
    display_images(titles, images)
'''
# Process all images
image_folder = 'THE3_Images/'
image_paths = [image_folder + str(i) + '.png' for i in range(1,6)]
for path in image_paths:
    process_image(path)
'''
#MAIN
if __name__ == '__main__':
    #Image5
    img_5 = read_image('5.png')
    image = img_5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = get_channel(image, 'red')
    # Apply KMeans with RGB
    kmeans_rgb_result = kmeans_rgb(image, k=6)
    write_image(kmeans_rgb_result, '5_kmeans_rgb.png')
    # Apply KMeans with LBP
    
    kmeans_lbp_result = kmeans_lbp(image, k=2, radius=2, n_points=16)
    write_image(kmeans_lbp_result, '5_kmeans_lbp.png')
    #Image6
