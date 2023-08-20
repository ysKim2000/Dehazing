'''
1. Dark channel features (DS)
2. MSCN features (MS)
3. Gradient features (GS)
4. ChromaHSV features (CS)

5. Fog(FD) 측정(DS*MS)
6. Artifacts(AD) 측정 (GS*CS)

7. 최종 품질 측정(결합) - (FD^{beta1}*AD^{beta2})
'''
import numpy as np
import cv2
from os import path as ospath
from scipy import signal

BINS = 256
MAX_LEVEL = BINS - 1

# Parameter setting
K1 = 0.0001
K2 = 0.00005
K3 = 0.00045
K4 = 0.0009
L = 255  # pixel value


def i2f(img):
    return img / MAX_LEVEL


def f2i(img):
    return np.uint8(np.around(img * MAX_LEVEL))


def C(K):
    return (K * L) ** 2


# 1. dark channel similarity
def DS(R, D):
    Rdc = np.min(i2f(R), 2)  # Reference image dark channel
    Ddc = np.min(i2f(D), 2)  # Defogged image dark channel

    CD = C(K1)

    # Dark channel similarity
    SD = (2 * Ddc * Rdc + CD) / (Rdc**2 + Ddc**2 + CD)
    return np.mean(SD), SD


# 2. MSCN similarity
def MS(R, D):
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY) / MAX_LEVEL # Reference image
    Dg = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY) / MAX_LEVEL # Defogged image

    R_MSCN = calculate_mscn_coefficients(Rg) 
    D_MSCN = calculate_mscn_coefficients(Dg)

    CM = C(K2)

    # MSCN similarity
    SM = (2 * R_MSCN * D_MSCN + CM) / (R_MSCN ** 2 + D_MSCN ** 2 + CM)

    return np.mean(SM), SM


# https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * \
        np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)


def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)

    return (image - local_mean) / (local_var + C)


# 3. gradient similarity
def GS(R, D):
    R_gradient = grad(R) # reference image gradient
    D_gradient = grad(D) # Defogged image gradient

    CG = C(K3)
    SG = (2*R_gradient*D_gradient+CG)/(D_gradient ** 2 + R_gradient ** 2 + CG)
    return np.mean(SG), SG

def grad(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / MAX_LEVEL
    # 수직, 수평 sobel
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_img = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad_img

# 4. color similarity
def CS(R, D):
    R_color = get_saturation(R) * get_value(R)    # reference image chroma_hsv
    D_color = get_saturation(D) * get_value(D)    # defogged image chroma_hsv

    CC = C(K4)
    SC = (2*R_color*D_color+CC)/(R_color ** 2 + D_color ** 2 + CC)
    return np.mean(SC), SC


def get_value(img):
    return np.average(img, 2)


def get_saturation(img):
    return 1.0 - np.min(img, 2) / (np.average(img, 2) + np.finfo(np.float32).eps)


def FRFSIM(R, D):
    # similarity features
    mean_SD, SD = DS(R, D)
    mean_SM, SM = MS(R, D)
    mean_SG, SG = GS(R, D)
    mean_SC, SC = CS(R, D)
    print(" SM:", mean_SM, " SD:", mean_SD, " SC:",mean_SC, " SG:",mean_SG)


    cv2.imwrite('assessment/result/dark_similarity.jpg', SD*255)
    cv2.imwrite('assessment/result/MSCN_similarity.jpg', SM*255)
    cv2.imwrite('assessment/result/gradient_similarity.jpg', SG*255)
    cv2.imwrite('assessment/result/color_similarity.jpg', SC*255)

    if 0.85 < mean_SD and mean_SD <= 1:
        beta1 = 0.2
        beta2 = 0.8
    elif 0 <= mean_SD and mean_SD <= 0.85:
        beta1 = 0.8
        beta2 = 0.2
    else:
        beta1 = 0.5
        beta2 = 0.5

    S_FD = mean_SD * mean_SM
    S_AD = mean_SG * mean_SC


    FRFSIM_MAP = S_FD**beta1 * S_AD**beta2
    # FRFSIM_MAP = np.power(S_FD, beta1) * np.power(S_AD, beta2)
    # FRFSIM_MAP = (np.sign(S_FD) * (np.abs(S_FD)) ** (beta1)) * \
    #     (np.sign(S_AD) * (np.abs(S_AD)) ** (beta2))

    cv2.imwrite('assessment/result/FRFSIM_MAP.jpg', FRFSIM_MAP*255)

    FRFSIM_VALUE = abs(np.mean(FRFSIM_MAP))
    print(FRFSIM_VALUE)


if __name__ == "__main__":
    DIR = 'outputs'
    # 0: clear image, 1: slightly image, 2: moderately, 3: highly image, 4: extremely image
    R_PATH = ospath.join('assessment', 'images', '0001-0.jpg')  # reference image
    D_PATH = ospath.join('assessment', 'images', '0001-2.jpg')  # target image

    R = cv2.imread(R_PATH)
    D = cv2.imread(D_PATH)

    FRFSIM(R, D)

    cv2.waitKey()
    cv2.destroyAllWindows()
