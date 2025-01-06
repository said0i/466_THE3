# Student1 Name: Ahmed Said Gençkol Student1 ID: 2539377
# Student2 Name: Batuhan Teberoğu Student2 ID: 2581056

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
#from sklearn.cluster import KMeans

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
    cv2.imwrite(output_folder+filename, img)
#Create necessary folders
def create_necessary_folders():
    folders = ["image1", "image2", "image3", "image4", "image5", "image6"]
    # Create the folders
    for folder in folders:
        folder_path = os.path.join(output_folder, folder)
        os.makedirs(folder_path, exist_ok=True)


#######################################
####//////BASIC FUNCTIONS\\\\\\\\######
#######################################
#Apply Kmeans clustering to an image
def kmeans_clustering(image, k=2):
    
    # convert to np.float32
    Z = np.float32(image)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    
    return res
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
'''
Method to determine the pattern:

        ``default``
            Original local binary pattern which is grayscale invariant but not
            rotation invariant.
        ``ror``
            Extension of default pattern which is grayscale invariant and
            rotation invariant.
        ``uniform``
            Uniform pattern which is grayscale invariant and rotation
            invariant, offering finer quantization of the angular space.
            For details, see [1]_.
        ``nri_uniform``
            Variant of uniform pattern which is grayscale invariant but not
            rotation invariant. For details, see [2]_ and [3]_.
        ``var``
            Variance of local image texture (related to contrast)
            which is rotation invariant but not grayscale invariant.
'''
def ELBP_skimage(src, radius, neighbors):
    lbp = local_binary_pattern(src, neighbors, radius, method='default')
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

#Morphological operations
def grayscale_morphology(image, operation, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    else:
        return cv2.morphologyEx(image, operation, kernel)

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
    pixels = image.reshape(-1, 3)
    #kmeans = KMeans(n_clusters=k, random_state=0).fit(image)
    kmeans = kmeans_clustering(pixels, k)
    #segmented = kmeans.labels_.reshape(image.shape[:2])
    res = kmeans.reshape(image.shape)
    return res

# KMeans using LBP Features (skimage)
def kmeans_lbp_skimage(image, k=2, radius=2, n_points=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = get_channel(image, 'red')
    lbp= ELBP_skimage(gray, radius, n_points)
    # Normalize the LBP values to the range [0, 256]
    lbp_normalized = cv2.normalize(lbp, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lbp_flattened = lbp_normalized.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=k, random_state=0).fit(lbp_flattened)
    kmeans = kmeans_clustering(lbp_flattened, k)
    segmented = kmeans.reshape(gray.shape)
    return segmented
# Another way of implementing kmeans with extended lbp
def kmeans_lbp(img, K, radius, neighbors):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_features = ELBP(gray_img, radius, neighbors)

    lbp_flattened = lbp_features.flatten().astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, label, center = cv2.kmeans(lbp_flattened, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((lbp_features.shape[0], lbp_features.shape[1]))
    return res 
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
    kmeans_lbp_result = kmeans_lbp_skimage(image, k=2, radius=2, n_points=8)

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
def image1_and_image2(img1, img2):
    # Image 1
    # Gray-scale Morphology
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Apply Morphological Gradient
    grad_1 = grayscale_morphology(gray_1, cv2.MORPH_GRADIENT, 13)
    # Postprocessing, thresholding
    _, final_1 = cv2.threshold(grad_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    write_image(final_1, 'image1/1_morph_gradient.png')

    # KMeans Clustering using RGB Features
    image1_kmeans = kmeans_rgb(img1, 2)
    write_image(image1_kmeans, 'image1/1_kmeans_rgb.png')

    # KMeans Clustering using Local Binary Pattern (LBP) Features
    image1_kmeans_lbp = kmeans_lbp(img1, 2, 2, 16)
    write_image(image1_kmeans_lbp, 'image1/1_kmeans_lbp.png')

    # Image 2
    # Gray-scale Morphology
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Apply Morphological Gradient
    morph_2 = grayscale_morphology(gray_2,  cv2.MORPH_GRADIENT, 2)
    # Postprocessing, thresholding
    _, morph_2 = cv2.threshold(morph_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    write_image(morph_2, 'image2/2_morph_gradient.png')

    # KMeans Clustering using RGB Features
    image2_kmeans = kmeans_rgb(img2, 2)
    write_image(image2_kmeans, 'image2/2_kmeans_rgb.png')

    # KMeans Clustering using Local Binary Pattern (LBP) Features
    image2_kmeans_lbp = kmeans_lbp(img2, 2, 2, 16)
    write_image(image2_kmeans_lbp, 'image2/2_kmeans_lbp.png')

def image3_and_image4(img3 ,img4):
    # Image 3
    # Gray-scale Morphology
    image3_blackhat = grayscale_morphology(img3, cv2.MORPH_BLACKHAT, 11)
    write_image(image3_blackhat, 'image3/3_blackhat.png')
    image3_gradient = grayscale_morphology(img3, cv2.MORPH_GRADIENT, 3)
    write_image(image3_gradient, 'image3/3_gradient.png')

    # KMeans Clustering using RGB Features
    image3_kmeans = kmeans_rgb(img3, 3)
    write_image(image3_kmeans, 'image3/3_kmeans_rgb.png')

    # KMeans Clustering using Local Binary Pattern (LBP) Features
    image3_kmeans_lbp = kmeans_lbp(img3, 2, 2, 8)
    write_image(image3_kmeans_lbp, 'image3/3_kmeans_lbp.png')

    # Image 4
    image4_blackhat = grayscale_morphology(img4, cv2.MORPH_BLACKHAT, 11)
    write_image(image4_blackhat, 'image4/4_blackhat.png')
    image4_gradient = grayscale_morphology(img4, cv2.MORPH_GRADIENT, 5)
    write_image(image4_gradient, 'image4/4_gradient.png')

    # KMeans Clustering using RGB Features
    image4_kmeans = kmeans_rgb(img4, 2)
    write_image(image4_kmeans, 'image4/4_kmeans_rgb.png')

    # KMeans Clustering using Local Binary Pattern (LBP) Features
    image4_kmeans_lbp = kmeans_lbp(img4, 2, 2, 16)
    write_image(image4_kmeans_lbp, 'image4/4_kmeans_lbp.png')

def image5_and_image6(img5, img6):
    #Image5
    gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    red = get_channel(img5, 'red')
    
   
    # Image 5
    # Gray-scale Morphology
    image5_blackhat = grayscale_morphology(img5, cv2.MORPH_BLACKHAT, 11)
    write_image(image5_blackhat, 'image5/5_blackhat.png')
    image5_gradient = grayscale_morphology(img5, cv2.MORPH_GRADIENT, 3)
    write_image(image5_gradient, 'image5/5_gradient.png')

    # Apply KMeans Clustering with RGB
    kmeans_rgb_result = kmeans_rgb(img5, k=2)
    #Apply postprocessing to make it black and white
    gray_image = cv2.cvtColor(kmeans_rgb_result, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    write_image(bw_image, 'image5/5_kmeans_rgb_threshholded.png')
    #write_image(kmeans_rgb_result, 'image5/5_kmeans_rgb.png')
    
    # Apply KMeans with LBP
    
    kmeans_lbp_result = kmeans_lbp_skimage(img5, k=2, radius=2, n_points=16)
    write_image(kmeans_lbp_result, 'image5/5_kmeans_lbp.png')
    # Image 6
    # Gray-scale Morphology
    image6_blackhat = grayscale_morphology(img6, cv2.MORPH_BLACKHAT, 11)
    write_image(image6_blackhat, 'image6/6_blackhat.png')
    image6_gradient = grayscale_morphology(img6, cv2.MORPH_GRADIENT, 3)
    write_image(image6_gradient, 'image6/6_gradient.png')

    # KMeans Clustering using RGB Features
    image6_kmeans = kmeans_clustering(img6, k=6)
    write_image(image6_kmeans, 'image6/6_kmeans_rgb_1.png')
     # Apply KMeans with RGB
    kmeans_rgb_result = kmeans_rgb(img6, k=2)
    write_image(kmeans_rgb_result, 'image6/6_kmeans_rgb_2.png')


    # KMeans Clustering using Local Binary Pattern (LBP) Features
    image6_kmeans_lbp = kmeans_lbp_skimage(img6,k=2, radius=2, n_points=16)
    write_image(image6_kmeans_lbp, 'image6/6_kmeans_lbp.png')

#MAIN
if __name__ == '__main__':
    
    # Create folders inside output folder to make it organized
    create_necessary_folders()
    # Read images
    img_1 = read_image('1.png')
    img_2 = read_image('2.png')
    img_3 = read_image('3.png')
    img_4 = read_image('4.png')
    img_5 = read_image('5.png')
    img_6 = read_image('6.png')

    # Images 1 and 2
    image1_and_image2(img_1, img_2)

    # Images 3 and 4
    #image3_and_image4(img_3, img_4)

    # Images 5 and 6
    #image5_and_image6(img_5, img_6)
    '''
    #Image5
    img_5 = read_image('5.png')
    image = img_5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = get_channel(image, 'red')
    # Apply KMeans with RGB
    kmeans_rgb_result = kmeans_rgb(image, k=6)
    write_image(kmeans_rgb_result, '5_kmeans_rgb.png')
    # Apply KMeans with LBP
    
    kmeans_lbp_result = kmeans_lbp_skimage(image, k=2, radius=2, n_points=16)
    write_image(kmeans_lbp_result, '5_kmeans_lbp.png')
    #Image6
    '''
