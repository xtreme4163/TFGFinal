import SwarmPackagePy
import cv2
from SwarmPackagePy import misfunciones as func
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
import os

parser= argparse.ArgumentParser()

#Agrego argumento de la imagen
parser.add_argument('imagen', type=str, help="Nombre de la imagen a procesar")
args = parser.parse_args()

# Obtener la ruta completa de la imagen
ruta_imagen = os.path.join(os.path.dirname(__file__), 'images', args.imagen)

# Verificar si el archivo existe
if not os.path.isfile(ruta_imagen):
    print(f"Error: No se puede encontrar el archivo '{ruta_imagen}'")
    quit()

# Intentar leer la imagen con OpenCV
img = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
if img is None:
    print(f"Error: No se puede abrir o leer el archivo '{ruta_imagen}'")
    quit()


# PARA EJECUTAR PSO, LUCIERNAGAS Y WSA, descomentar el que se quiera probar, 
#PSO
# Argumentos ( nº particulas, funcion, LIM_MIN, LIM_MAX, dimension, iteraciones, inercia, c1,c2)
alh = SwarmPackagePy.pso(8, func.getMse, 0, 255, 3, 10, w=0.729, c1=2.05, c2=2.05, imagen=ruta_imagen)
                         
#Luciernagas
#alh = SwarmPackagePy.fa(5, func.getMae, 0, 255, 3, 10, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1)

#Ballenas
#alh = SwarmPackagePy.ballena(10, func.getMse, 0, 255, 3, 10, ro0=2, eta=0.005)

#Lobos
#alh = SwarmPackagePy.gwo(30, func.getMse, 0, 255, 3, 30)

#Abejas
#alh=SwarmPackagePy.aba(30, func.getMse, 0, 255, 3, 30)


