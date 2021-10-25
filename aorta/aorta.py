import cv2
import os
from matplotlib import pyplot as plt
import argparse
import time
import json
from preprocessing import *
import pdb

### Globales ------------------------------------------------
nombre_imagenes = []  # nombre imágenes
imagenes = []         # imágenes np.array
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
    imagen = cv2.imread(nombre_img,cv2.IMREAD_GRAYSCALE)
    inicio = time.perf_counter() 

    # ETAPA 1 -------------------------------------------------------
    resultado = preprocesar(imagen)

    # ETAPA 2 -------------------------------------------------------
    #resultado = cv2.GaussianBlur(resultado, (5,5), 0)
    #resultado = cv2.Canny(resultado,100,200)  # 5 30 
    
    fin = time.perf_counter()
    tiempo += (fin-inicio)

    # ALMACENAR RESULTADOS
    imagenes.append(imagen)
    resultados.append(resultado)

    print("%s |%s%s| %d/%d [%d%%] in %.2fs (eta: %.2fs)"  % ("Processing...",u"\u2588" * i," " * (len(imagenes)-i),i,len(imagenes),int(i/len(imagenes)*100),tiempo,(fin-inicio)*(len(imagenes)-i)),end='\r', flush=True)
    i+=1


# SALVAR RESULTADOS

###### Versión rápida
i=1
for nombre, img, resultado in zip(nombre_imagenes, imagenes, resultados):
    cv2.imwrite(os.path.join(output_directory,os.path.basename(nombre)+".png"), resultado) 
    i+=1
