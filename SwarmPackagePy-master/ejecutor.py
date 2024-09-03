import SwarmPackagePy
import cv2
from SwarmPackagePy import misfunciones as func
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
import os


#Para instalar librerias -> pip install -r requirements.txt

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


#individuos=15
#iteraciones=10


def matchFuncionPso(args, ruta_imagen, individuos, iteraciones):
    match(args.funcion):
        case "MSE":
            alh = SwarmPackagePy.pso(individuos, func.getMse, 0, 255, 3, iteraciones, args.numeroColores,args.pintaImagen,w=0.729, c1=2.05, c2=2.05, imagen=ruta_imagen)
        case "MAE":
            alh = SwarmPackagePy.pso(individuos, func.getMae, 0, 255, 3, iteraciones, args.numeroColores,args.pintaImagen,w=0.729, c1=2.05, c2=2.05, imagen=ruta_imagen)
        case "SSIM":
            alh = SwarmPackagePy.pso(individuos, func.getSsim, 0, 255, 3, iteraciones, args.numeroColores,args.pintaImagen,w=0.729, c1=2.05, c2=2.05, imagen=ruta_imagen)
        case "MSSIM":
            alh = SwarmPackagePy.pso(individuos, func.getMsSsim, 0, 255, 3, iteraciones, args.numeroColores,args.pintaImagen,w=0.729, c1=2.05, c2=2.05, imagen=ruta_imagen)

def matchFuncionFa(args, ruta_imagen, individuos, iteraciones):
    match(args.funcion):
        case "MSE":
             alh = SwarmPackagePy.fa(individuos, func.getMse, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1,imagen=ruta_imagen)
        case "MAE":
             alh = SwarmPackagePy.fa(individuos, func.getMae, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1,imagen=ruta_imagen)
        case "SSIM":
             alh = SwarmPackagePy.fa(individuos, func.getSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1,imagen=ruta_imagen)
        case "MSSIM":
             alh = SwarmPackagePy.fa(individuos, func.getMsSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1,imagen=ruta_imagen)

def matchFuncionBa(args, ruta_imagen, individuos, iteraciones):
    match(args.funcion):
        case "MSE":
            alh = SwarmPackagePy.ballena(individuos, func.getMse, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, ro0=2, eta=0.005,imagen=ruta_imagen)
        case "MAE":
            alh = SwarmPackagePy.ballena(individuos, func.getMae, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, ro0=2, eta=0.005,imagen=ruta_imagen)
        case "SSIM":
            alh = SwarmPackagePy.ballena(individuos, func.getSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, ro0=2, eta=0.005,imagen=ruta_imagen)
        case "MSSIM":
            alh = SwarmPackagePy.ballena(individuos, func.getMsSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen, ro0=2, eta=0.005,imagen=ruta_imagen)

def matchFuncionGwo(args, ruta_imagen, individuos, iteraciones):
    match(args.funcion):
        case "MSE":
            alh = SwarmPackagePy.gwo(individuos, func.getMse, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "MAE":
            alh = SwarmPackagePy.gwo(individuos, func.getMae, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "SSIM":
            alh = SwarmPackagePy.gwo(individuos, func.getSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "MSSIM":
            alh = SwarmPackagePy.gwo(individuos, func.getMsSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)

def matchFuncionAba(args, ruta_imagen, individuos, iteraciones):
    match(args.funcion):
        case "MSE":
            alh=SwarmPackagePy.aba(individuos, func.getMse, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "MAE":
            alh=SwarmPackagePy.aba(individuos, func.getMae, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "SSIM":
            alh=SwarmPackagePy.aba(individuos, func.getSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)
        case "MSSIM":
            alh=SwarmPackagePy.aba(individuos, func.getMsSsim, 0, 255, 3, iteraciones,args.numeroColores,args.pintaImagen,imagen=ruta_imagen)



match args.algoritmo:
    case "PSO":
        #PSO
        matchFuncionPso(args, ruta_imagen, args.individuos, args.iteraciones)
    case "FA":
        #Luciernagas
        matchFuncionFa(args, ruta_imagen, args.individuos, args.iteraciones)
    case "BA":
        #Ballenas
        matchFuncionBa(args, ruta_imagen, args.individuos, args.iteraciones)
    case "GWO":
        #Lobos
        matchFuncionGwo(args, ruta_imagen, args.individuos, args.iteraciones)       
    case "ABA":
        #Abejas
        matchFuncionAba(args, ruta_imagen, args.individuos, args.iteraciones)       
    case _:
            print("Algoritmo no reconocido")


