import SwarmPackagePy
import cv2
from SwarmPackagePy import misfunciones as func
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
import os

import time

#Para instalar librerias -> pip install -r requirements.txt


# Función que ejecuta el algoritmo correspondiente
def ejecutar_algoritmo(algoritmo, funcion, individuos, iteraciones, numero_colores, imagen):
    if algoritmo not in algoritmos:
        print(f"Error: Algoritmo '{algoritmo}' no reconocido")
        quit()

    if funcion not in funcionesObjetivo:
        print(f"Error: Funcion '{funcion}' no reconocida")
        quit()

    # Ejecutar el algoritmo
    alg_func = algoritmos[algoritmo]
    func_fitness = funcionesObjetivo[funcion]
    alg_func(individuos, func_fitness, numero_colores, imagen, iteraciones)


parser= argparse.ArgumentParser()

#Agrego argumento de la imagen
parser.add_argument('imagen', type=str, help="Nombre de la imagen a procesar")
parser.add_argument('numeroColores', type=int, help="Numero de colores de la nueva imagen")
parser.add_argument('algoritmo', type=str, help="Algoritmo a procesar")
parser.add_argument('funcion', type=str, help="Funcion a usar por algoritmo (ej. MSE)")
parser.add_argument('iteraciones', type=int, help="Numero de iteraciones que realizara el algoritmo")
parser.add_argument('individuos', type=int, help="Numero de individuos del algoritmo")
parser.add_argument('--pintaImagen', type=bool, default=False,help="Argumento para saber si pinta imagen al final del algoritmo (depuracion); Si viene se pinta.")
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

#DICCIONARIOS FUNCIONES ALGORITMOS
#En este punto se pueden añadir nuevas funciones y algoritmos. En los algoritmos solo hace falta definir los argumentos que se le pasan
# Definir funciones disponibles para el algoritmo
funcionesObjetivo = {
    "MSE": func.getMse,
    "MAE": func.getMae,
    "SSIM": func.getSsim,
    "MSSIM": func.getMsSsim
}

# Definir los algoritmos disponibles
algoritmos = {
    "PSO": lambda indiv, func, col, img, it: SwarmPackagePy.pso(indiv, func, 0, 255, 3, it, col, args.pintaImagen, w=0.729, c1=2.05, c2=2.05, imagen=img),
    "FA": lambda indiv, func, col, img, it: SwarmPackagePy.fa(indiv, func, 0, 255, 3, it, col, args.pintaImagen, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1, imagen=img),
    "BA": lambda indiv, func, col, img, it: SwarmPackagePy.ballena(indiv, func, 0, 255, 3, it, col, args.pintaImagen, ro0=2, eta=0.005, imagen=img),
    "GWO": lambda indiv, func, col, img, it: SwarmPackagePy.gwo(indiv, func, 0, 255, 3, it, col, args.pintaImagen, imagen=img),
    "ABA": lambda indiv, func, col, img, it: SwarmPackagePy.abejas(indiv, func, 0, 255, 3, it, col, args.pintaImagen, imagen=img),
    "WOA": lambda indiv, func, col, img, it: SwarmPackagePy.abejas(indiv, func, 0, 255, 3, it, col, args.pintaImagen, imagen=img)
}

# Ejecutar el algoritmo solicitado
ejecutar_algoritmo(args.algoritmo, args.funcion, args.individuos, args.iteraciones, args.numeroColores, ruta_imagen)
