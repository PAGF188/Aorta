import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
from skimage.morphology import flood
from skimage.morphology import thin

def paredes(img):
    """
    - Localizar las paredes de la aorta.
    - Generar máscara del interior de la aorta.
    
    Parameters
    ----------
    img : numpy.ndarray | imagen GRAY 
    
    Returns
    -------
    paredes : numpy.ndarray | mascara binaria 0 o 255
    borde :   numpy.ndarray | mascara binaria 0 o 255
    """

    # Localizamos la componente conexa del píxel central (pixel interior aorta).
    r,w = img.shape
    mask = flood(img, (r//2, w//2))
    paredes = img * 1
    paredes[mask] = 255
    ix,iy = np.where(paredes!=255)
    paredes[ix,iy]=0

    # Obtenemos su borde
    borde = cv2.Canny(paredes,1000,1400)

    return paredes, borde

def get_aortic_params(pared, borde_pared):
    """
    
    Parameters
    ----------
    pared : numpy.ndarray | mascara binaria 0 o 255 | localización interior aorta.
    borde_pared : numpy.ndarray | mascara binaria 0 o 255 | localización solo paredes.
    
    Returns
    -------
    area: int | nº de píxeles que forman la aorta.
    centro: numpy.ndarray (2,) | int(media) de todos los puntos que integran la aorta.
    radio: float | media distancias de cada punto de borde al centro.
    circularidad: float [0-1] | media diferencias radio "real" radio "esperado" relativizado. 
    """

    indices_ = np.where(pared==255) # indices píxeles aorta
    indices_borde = np.where(borde_pared==255) # indice píxeles borde aorta

    # AREA ------> nº de píxeles que integran aorta (blancos)
    area = len(indices_[0]) 

    # CENTRO ----> media de todos los puntos que integran la aorta (blancos)
    centro = (np.sum(indices_,axis=1) / area).astype(int)

    # RADIO -----> para mejor estimación usamos la distancia media de cada punto de borde al centro
    puntos_borde = np.array(indices_borde).T
    distancias_centro_borde = [np.linalg.norm(centro - aux) for aux in puntos_borde]
    radio = np.sum(distancias_centro_borde)/len(distancias_centro_borde)

    # CIRCULARIDAD --> idealmente en un círculo perfecto el radio es el mismo para todos los
    #                  puntos del borde. Es decir distancias_centro_borde - radio tendría que ser
    #                  0 para todos los puntos del borde. Grado de circularidad calculado
    #                  como la diferencia media entre distancias_centro_borde y radio (y
    #                  relativizado).
    diferencia_m = np.sum([np.abs(radio-aux) for aux in distancias_centro_borde]) / len(distancias_centro_borde)
    circularidad = (radio - diferencia_m)/radio
    
    return area,centro,radio,circularidad

def stents(img_preprocesada, centro, radio, borde_pared):
    RAD_MAX = 50  # VALOR A AJUSTAR
    # Nos quedamos solo con radio + RAD_MAX
    mascara = img_preprocesada * 0
    mascara = cv2.circle(mascara,centro[::-1],int(radio + RAD_MAX),1,cv2.FILLED)
    aux = img_preprocesada * mascara

    # Círculo exterior
    aux = cv2.circle(aux,centro[::-1],int(radio+RAD_MAX),1,thickness=3)
    
    # "Círculo" (no es un circulo) interior
    c1 = cv2.resize(borde_pared, (0,0), fx=1.23, fy=1.23)
    ii,jj = np.where(c1>0) 
    c1[ii,jj] = 1
    diferencia = np.array(c1.shape) - np.array(img_preprocesada.shape)
    c1 = c1[diferencia[0]//2 : -diferencia[0]//2,diferencia[1]//2 : -diferencia[1]//2]
    c1 = thin(c1)
    
    # borrar -> para pintar ambos
    # ii,jj = np.where(c1>0) 
    # aux[ii,jj] = 0


    # Buscamos corte entre ambos
    c1 = 1-c1
    cortes = np.logical_or(aux,c1)
    ii,jj = np.where(cortes==0)

    n_region = 2
    for corte in zip(ii,jj):
        if aux[corte] == 0:
            region = flood(aux, corte)
            # Si la región no supera el limite
            if np.count_nonzero(region)<3000:
                aux[region] = n_region
                n_region+=1
    
    print("Numero de regiones:", n_region-2)
    plt.imshow(aux)
    plt.show()
   
    
    
    return aux