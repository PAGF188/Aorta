import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt


def step1(img):
    """
    Paso 1 del preprocesamiento. Truncamiento basado en posición:
    Eliminar:
        - Subimagen inferior
        - Exterior 1º círculo
        - Círculo interno
    
    Parameters
    ----------
    img : numpy.ndarray | imagen GRAY 
    
    Returns
    -------
    img : numpy.ndarray | imagen GRAY 
    """
    TRUNCAMIENTO_ = 528  # Eliminar subimagen inferior
    RADIO1_ = 262        # Eliminar exterior 1º círculo
    RADIO2_ = 55         # Eliminar círculo interno

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
    Paso 2 del preprocesamiento. Truncamiento basado en intensidad:
    - Eliminar líneas blancas
    - Binarizar la imagen en zona de interés.
    
    Parameters
    ----------
    img : numpy.ndarray | imagen GRAY
    
    Returns
    -------
    img : numpy.ndarray | imagen binaria | 0-1
    """

    UMBRAL_O = 25   # Binarización
    UMBRAL_C = 225  # Eliminar líneas blancas

    # Thresholding
    ix,iy = np.where(img>=UMBRAL_C)
    img[ix,iy]=0

    ix,iy = np.where(img<=UMBRAL_O)
    img[ix,iy]=0

    ix,iy = np.where(img>UMBRAL_O)
    img[ix,iy]=1  # 255

    # Depuración resultados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=2)

    kernel2 = np.zeros((5,3))
    kernel2[:,1] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2.astype(np.uint8),iterations=1)

    return img

def step3(img):  # MSER -> descartado
    MIN_A = 1
    MAX_A = 10
    DELTA_ = 10
    aux = img*1
    rows,cols = img.shape
    blurred = cv2.blur(img,(5,5))
    min=int(MIN_A/100*rows*cols)
    max=int(MAX_A/100*rows*cols)
    mser = cv2.MSER_create(DELTA_,min,max,0.8)
    regions, bboxes = mser.detectRegions(blurred)   
    for r in regions:
        i,j=np.transpose(np.fliplr(r))
        aux[i,j]=(255)
    return aux

def preprocesar(image):
    """
    Preprocesamiento de la imagen.
    
    Parameters
    ----------
    img : numpy.ndarray | imagen GRAY
    
    Returns
    -------
    img : numpy.ndarray | imagen binaria | 0-1 | ROI
    """
    return step2(step1(image))
