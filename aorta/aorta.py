import cv2
import os
from matplotlib import pyplot as plt
import argparse
import time
import json
from preprocessing import *
from processing import *
import pdb

### Globales ------------------------------------------------
nombre_imagenes = []  # nombre imágenes
imagenes = []         # imágenes np.array
preprocesadas = []    # imagenes preprocesadas
resultados = []       # imagenes procesadas
tiempo = 0
### ---------------------------------------------------------


# Parser --list
#     - Si es archivo -> almacenar para procesar.
#     - Si es directorio -> explorar y almacenar sus archivos para procesar (solo 1 nivel).
# Parser --output
#     - Si no existe lo creamos.
# Parser --evaluar
#     - Si se pasa archivo anotaciones json evaluamos etiquetas predichas con reales.

parser = argparse.ArgumentParser(description='Automatic OCT processing')
parser.add_argument('-l','--list', nargs='+', help='<Required> Images to process', required=True)
parser.add_argument('-o','--output', help='<Required> Place to save results', required=True)
args = parser.parse_args()

args = parser.parse_args()

for element in args.list:
    if os.path.isfile(element):
        nombre_imagenes.append(element)
    elif os.path.isdir(element):
        for f in os.listdir(element):
            if os.path.isfile(os.path.join(element,f)):
                nombre_imagenes.append(os.path.join(element,f))

output_directory = args.output
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# Procesamiento de cada imagen
print("%d %s" %(len(nombre_imagenes), "images are going to be processed...\n"))
print("%s |%s%s| %d/%d [%d%%] in %.2fs"  % ("Processing...","-" * 0," " * (len(nombre_imagenes)-0),0,len(nombre_imagenes),0,0),end='\r', flush=True)

i=1
for nombre_img in nombre_imagenes:
    print(nombre_img)
    imagen = cv2.imread(nombre_img,cv2.IMREAD_GRAYSCALE)
    inicio = time.perf_counter() 

    # ETAPA 1: Preprocesar ------------------------------------------
    img_preprocesada = preprocesar(imagen)

    # ETAPA 2: Limitación aorta + params ----------------------------
    pared, borde_pared = paredes(img_preprocesada)
    aortic_params = get_aortic_params(pared, borde_pared)
    print(aortic_params)

    # ETAPA 3: Stents -----------------------------------------------
    r = stents(img_preprocesada, aortic_params[1], aortic_params[2], borde_pared)

    fin = time.perf_counter()
    tiempo += (fin-inicio)


    # ALMACENAR RESULTADOS
    imagenes.append(imagen)
    preprocesadas.append(img_preprocesada*255)
    resultados.append(r)

    #print("%s |%s%s| %d/%d [%d%%] in %.2fs (eta: %.2fs)"  % ("Processing...",u"\u2588" * i," " * (len(imagenes)-i),i,len(imagenes),int(i/len(imagenes)*100),tiempo,(fin-inicio)*(len(imagenes)-i)),end='\r', flush=True)
    i+=1


# SALVAR RESULTADOS
###### Versión rápida
i=1
for nombre, img, prepo, resultado in zip(nombre_imagenes, imagenes, preprocesadas, resultados):
    cv2.imwrite(os.path.join(output_directory,os.path.basename(nombre)+".png"), resultado) 
    i+=1
