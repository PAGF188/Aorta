import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt


def step1(img):
    """
    Paso 1 del preprocesamiento de la imagen:
    Eliminar todo lo que no sea el círculo central
    
    Parameters
    ----------
    image : numpy.ndarray | imagen GRAY 
    
    Returns
    -------
    image : numpy.ndarray | imagen GRAY 
    """
    TRUNCAMIENTO_ = 528
    RADIO1_ = 262
    RADIO2_ = 55

    img = img[0:TRUNCAMIENTO_,:]
    sy,sx = img.shape
    x = np.linspace(-sx/2, sx/2,sx)
    y = np.linspace(-sy/2, sy/2,sy)
    x, y = np.meshgrid(x, y) 
    d = np.sqrt(x**2 + y**2)
    img[d>RADIO1_] = 0 
    img[d<RADIO2_] = 0
    return img

def step2(img):
    """
    Paso 2 del preprocesamiento de la imagen:
    Dentro del círculo central: 
    - Eliminar subcírculo interno
    - Eliminar líneas blancas
    
    Parameters
    ----------
    image : numpy.ndarray | imagen GRAY 
    
    Returns
    -------
    image : numpy.ndarray | imagen GRAY 
    """
    UMBRAL_O = 40
    UMBRAL_C = 225
    ix,iy = np.where(img<=UMBRAL_O)
    img[ix,iy]=0

    ix,iy = np.where(img>=UMBRAL_C)
    img[ix,iy]=0

    return img

def preprocesar(image):
    return step2(step1(image))
