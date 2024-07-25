import SwarmPackagePy
import cv2
#from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import misfunciones as func
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np






# PARA EJECUTAR PSO, LUCIERNAGAS Y WSA, descomentar el que se quiera probar, 
#PSO
# Argumentos ( nº particulas, funcion, LIM_MIN, LIM_MAX, dimension, iteraciones, inercia, c1,c2)
alh = SwarmPackagePy.pso(5, func.getMae, 0, 255, 3, 10, w=0.729, c1=2.05, c2=2.05)
                         
#Luciernagas
#alh = SwarmPackagePy.fa(5, func.getMae, 0, 255, 3, 10, csi=1, psi=1, alpha0=1, alpha1=0.1, norm0=0, norm1=0.1)

#Ballenas
#alh = SwarmPackagePy.ballena(10, func.getMse, 0, 255, 3, 10, ro0=2, eta=0.005)

#Lobos ?


