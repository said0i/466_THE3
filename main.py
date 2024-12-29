# Student1 Name: Ahmed Said Gençkol Student1 ID: 2539377
# Student2 Name: Batuhan Teberoğu Student2 ID: 2581056

import numpy as np
import cv2 
import os
import pywt
import numpy as np

input_folder = 'THE3_Images/'
output_folder = 'THE3_Outputs/'

def read_image(filename):
    img = cv2.imread(input_folder + filename, cv2.IMREAD_UNCHANGED)
    return img

def write_image(img, filename):
    cv2.imwrite(output_folder+filename, img)

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

if __name__ == '__main__':
    # Q1 - Segmentation
    img_1 = read_image('1.png')
    img_2 = read_image('2.png')
    img_3 = read_image('3.png')
    img_4 = read_image('4.png')
    img_5 = read_image('5.png')
    img_6 = read_image('6.png')

    # Images 1 and 2
    # gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    # morph_1 = morph(gray_1, (13,13), cv2.MORPH_GRADIENT)
    # _, morph_1 = cv2.threshold(morph_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morph_1 = morph(morph_1, (7,7), cv2.MORPH_OPEN)
    # write_image(morph_1, '1_morph_gradient.png')

    # gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    # morph_2 = morph(gray_2, (2,2), cv2.MORPH_GRADIENT)
    # _, morph_2 = cv2.threshold(morph_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morph_2 = morph(morph_2, (2,2), cv2.MORPH_OPEN)
    # write_image(morph_2, '2_morph_gradient.png')

    # Image 2 Kmeans
    # samples = img_2.reshape((-1,3))
    # samples = np.float32(samples)
    
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 2
    # _, label, center = cv2.kmeans(samples,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res = res.reshape((img_2.shape))
    # write_image(res, '2_kmeans.png')

    # Image 2 Kmeans with LBP
    # Extract LBP features
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    radius = 2
    neighbors = 16
    lbp_features = ELBP(gray_1, radius, neighbors)

    lbp_flattened = lbp_features.flatten().astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, label, center = cv2.kmeans(lbp_flattened, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((lbp_features.shape[0], lbp_features.shape[1]))
    write_image(res, f'1_kmeans_lbp_{radius}_{neighbors}.png')

    # Images 3 and 4 - Get wrinkle
    # write_image(morph(img_3, (11,11), cv2.MORPH_BLACKHAT), '3_blackhat_wrinkle.png')
    # write_image(morph(img_4, (11,11), cv2.MORPH_BLACKHAT), '4_blackhat_wrinkle.png')
    # write_image(morph(img_3, (3,3), cv2.MORPH_GRADIENT), '3_wrinkle.png')
    # write_image(morph(img_4, (5,5), cv2.MORPH_GRADIENT), '4_wrinkle.png')