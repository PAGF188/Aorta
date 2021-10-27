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
    plt.imshow(image_flooded)
    plt.show()
    return None