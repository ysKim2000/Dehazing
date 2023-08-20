# pip install opencv-python
import cv2

import os

from dcp import DCP
from cap import CAP
from sbte import SBTE


I_DIR = './data'
O_DIR = './outputs'

def file(filename):
    I_PATH = os.path.join(I_DIR, filename)
    
    I = cv2.imread(I_PATH)
    # cv2.imshow('Input', I)
    # cv2.imshow('DCP', DCP(I))
    # cv2.imshow('CAP', CAP(I) / 255)
    # cv2.imshow('SBTE', SBTE(I))

    cv2.imwrite(os.path.join(O_DIR, filename), I)
    cv2.imwrite(os.path.join(O_DIR, 'DCP_'+ filename), DCP(I))
    cv2.imwrite(os.path.join(O_DIR, 'CAP_'+ filename), CAP(I))
    cv2.imwrite(os.path.join(O_DIR, 'SBTE_'+ filename), SBTE(I))


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def files(dir=I_DIR):
    names = os.listdir(dir)

    for name in names:
        print(name)
        I = cv2.imread(os.path.join(dir, name))
        name = name.split('.')[0]
        cv2.imwrite(os.path.join(O_DIR, f'{name}.jpg'), I)
        cv2.imwrite(os.path.join(O_DIR, f'{name}_DCP.jpg'), DCP(I))
        cv2.imwrite(os.path.join(O_DIR, f'{name}_CAP.jpg'), CAP(I))
        cv2.imwrite(os.path.join(O_DIR, f'{name}_SBTE.jpg'), SBTE(I))


if __name__ == "__main__":
    file('GRCN_Google_346.jpg')
    # files()
