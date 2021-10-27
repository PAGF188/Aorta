import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
from skimage.morphology import flood

def paredes(img):
    r,w = img.shape
    mask = flood(img, (r//2, w//2))
    image_flooded = img.copy()
    image_flooded[mask] = 132
    ix,iy = np.where(image_flooded!=132)
    image_flooded[ix,iy]=0

    sobelx = cv2.Sobel(image_flooded, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_flooded, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
    return sobel