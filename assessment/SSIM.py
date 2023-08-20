# Usage:
#
# python3 script.py -f original.png -s modified.png
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

# 1. Import the necessary packages
#from skimage.measure import compare_ssim as ssim # deprecated (old version)
from skimage.metrics import structural_similarity as ssim 
import argparse
import imutils
import cv2
from skimage import io

# 2. Construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
# ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
# args = vars(ap.parse_args())

# 3. Load the two input images
# imageA = cv2.imread(args["first"])
# imageB = cv2.imread(args["second"])
imageA = cv2.imread('assessment/data/GRCN.png')
imageB = cv2.imread('assessment/data/sbte.jpg')

# cv2.imshow('A', imageA)
# cv2.imshow('B', imageB)

# 만약 url에서 불러올 거면 다음을 활용
#imageA = io.imread(args["first"])
#imageB = io.imread(args["second"])


# 4. Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 5. Compute the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# diff가 어떻게 되는지 볼 수 있습니다.
# cv2.imshow('diff',diff)
cv2.imwrite('assessment/result/diff_GRCN.jpg', diff)
# cv2.waitKey(0)

# 6. You can print only the score if you want
print("SSIM: {}".format(score))