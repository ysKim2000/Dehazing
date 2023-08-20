"""
Fast Single Image Dehazing Using Saturation Based Transmission Map Estimation
Original source
https://sites.google.com/view/ispl-pnu/software#h.y6sdafma686
Modified by Wonvin Kim, 09/04/22
"""

import cv2 
import numpy as np
import math
from os import path as ospath
from skimage import morphology

BINS = 256
MAX_LEVEL = BINS - 1


def i2f(img):
    return img / MAX_LEVEL

def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))


# Compute 'A' as described by Tang et al. (CVPR 2014)
def compute_A_Tang(img):
    h, w = img.shape[:2]
    
    dark_channel = morphology.erosion(np.min(img, 2), morphology.square(15))
    H, edges = np.histogram(dark_channel, 200)
    CDF = np.cumsum(H)
    idx = np.nonzero(CDF > h * w * 0.99)[0][0]
    threshold = edges[idx]
    mask = dark_channel >= threshold

    A = np.zeros(3)
    b, g, r = cv2.split(img)
    A[2] = np.median(r[mask])
    A[1] = np.median(g[mask])
    A[0] = np.median(b[mask])

    return A


# White balance using gray world assumption
def gray_world(img): # gray-world assumption algorithm
    b, g, r = cv2.split(img)

    R = np.empty(img.shape, img.dtype) 

    # r,g,b channeldml 평균값
    mu_r = np.average(r)
    mu_g = np.average(g)
    mu_b = np.average(b)
    
    # ?
    R[..., 0] = np.minimum(b * (mu_g / mu_b), 1.0)
    R[..., 2] = np.minimum(r * (mu_g / mu_r), 1.0)
    R[..., 1] = g
 
    return  R

def normalize(img):
    R = np.empty(img.shape, img.dtype) 

    for i in range(3):
        min = img[..., i].min()
        max = img[..., i].max()
        R[..., i] = (img[..., i] - min) / (max - min)
        R[..., i] = np.clip(R[..., i], 0.0, 1.0)     
        
    return R

def get_value(img):
    return np.average(img, 2)

def get_saturation(img):
    return 1.0 - np.min(img, 2) / (np.average(img, 2) + np.finfo(np.float32).eps)


# Estimate saturation of scene radiance: 3 Stretch methods
def estimate_saturation(saturation, p1): # First Method
    p2 = 2.0
    k1 = 0.5 *(1.0 - cv2.pow(1.0 - 2.0 * saturation, p1))
    k2 = 0.5 + 0.5 * cv2.pow((saturation - 0.5) / 0.5, p2)
    R = np.where(saturation <= 0.5, k1, k2)
    return np.maximum(R, saturation)

def estimate_saturation_quadratic(saturation): # Second Method
    return saturation * (2.0 - saturation)

def estimate_saturation_gamma(saturation, gamma): # Third Method
    R = (np.power(saturation, 1.0 / gamma) + 1.0 - np.power(1.0 - saturation, 1.0 / gamma)) / 2.0
    return np.maximum(R, saturation)

def estimate_TransimissionMap(value, i_saturation, j_saturation):
    R = 1.0 - value * (j_saturation - i_saturation) / j_saturation
    return np.clip(R, np.finfo(np.float32).eps, 1.0)

# map: transmission map, A: atmospheric light
def recover(img, map, A):
    J = np.empty(img.shape, img.dtype)
    
    for i in range(3):
        J[..., i] = (img[..., i] - A[i]) / map + A[i]
        J[..., i] = np.clip(J[..., i], 0.0, 1.0)
   
    return J

def adjust(img, percentile_low, percentile_high):
    R = np.empty(img.shape, img.dtype)

    low = np.percentile(img, percentile_low)
    high = np.percentile(img, percentile_high)

    for i in range(3):
        R[..., i] = (img[..., i] - low) / (high - low)
        R[..., i] = np.clip(R[..., i], 0.0, 1.0)
        
    return R

def clahe(img, clip):
  HSV = cv2.cvtColor(f2i(img), cv2.COLOR_BGR2HSV)
  clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
  HSV[..., 2] = clahe.apply(HSV[..., 2])
  return i2f(cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR))


# Saturation Based Transmission Map Estimation
def SBTE(img, is_only_result=True):
    I = i2f(img) # image float화
    A = compute_A_Tang(I) # A 값

    # Compute white balanced A
    I_WB = gray_world(I) # get White Blance I(Hazy image)
    A_WB = compute_A_Tang(I_WB)

    '''
    Parameter set
    - parameters for adjust: PER_LOW = 0.5, PER_HIGH = 99.9 (PER: percentile)
    - parameters for selecting two branch: EPSILON = (0.0, 0.1)
    - parameters for clahe: CLIP = 1 (0, 2)
    '''
    PER_HIGH = 99.9
    PER_LOW = 0.5
    EPSILON = 0.02
    CLIP = 1
    
    # Phase I - Normal
    if np.max(A) - np.min(A) < np.max(A_WB) - np.min(A_WB) + EPSILON:
        print('Phase I - Normal')
        phase = 1

    # Phase II - White Balance
    else: 
        print('Phase II - White Balance')
        phase = 2
        A = A_WB
        
    out = np.empty(I.shape, I.dtype) 
    for i in range(3):
        out[..., i] = I[..., i] / A[i]
    
    out = normalize(out)
    value = get_value(out)
    i_saturation = get_saturation(out)

    # j_saturation = estimate_saturation(i_saturation, 2.0)
    # j_saturation = estimate_saturation_quadratic(i_saturation)
    j_saturation = estimate_saturation_gamma(i_saturation, 0.2)

    transmission_map = estimate_TransimissionMap(value, i_saturation, j_saturation)
    J = recover(I, transmission_map, A)
    J = adjust(J, PER_LOW, PER_HIGH) # 0.5 ~ 99.9

    if phase == 2: J = gray_world(J) 
        
    J_enhanced = clahe(J, CLIP)
    J_enhanced = f2i(J_enhanced)
    J = f2i(J)

    return J if is_only_result else (J, J_enhanced, f2i(transmission_map))


if __name__ == "__main__":
    DIR = 'outputs'

    O_PATH = ospath.join(DIR, 'sbte.jpg')
    I_PATH = ospath.join('data', 'lake.jpg')

    I = cv2.imread(I_PATH)
    J, J_enhanced, transmission_map = SBTE(I, is_only_result=False)

    cv2.imwrite(O_PATH, J)
    cv2.imshow('Haze image', I)
    cv2.imshow('Dehazed image', J / MAX_LEVEL)

    cv2.imwrite(ospath.join(DIR, 'transmission map (SBTE).jpg'), transmission_map)
    cv2.imwrite(ospath.join(DIR, 'enhanced J (SBTE).jpg'), J_enhanced)

    cv2.waitKey()
    cv2.destroyAllWindows()
