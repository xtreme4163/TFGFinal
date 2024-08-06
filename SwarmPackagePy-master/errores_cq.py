# #####################################################################
# Código que calcula varias medidas de error para comparar 2 imágenes.
# Aplicado a cuantificación de color, compara la imagen original y la imagen cuantizada
#
# --------------------------------------------------------------------
# MEDIDAS DE ERROR QUE CALCULA:
# -- MSE
# -- MAE
# -- PSNR
# -- SSIM (realmente es el promedio para una ventana deslizante que recorre
#          la imagen)
# -- MS-SSIM
# --------------------------------------------------------------------
#
#  EJECUCIÓN: python3 errores_cq.py <img1> <img2>
#    <img1> e <img2> son dos ficheros de imagen. Se supone que representan la
#     misma imagen, pero con calidades distintas. 
#     El tamaño de ambas imágenes debe coincidir (filas y columnas)
#
# ULTIMA VERSION: 5-Mayo-2023
# María Luisa Pérez Delgado
# #####################################################################


import cv2
import numpy as np


# para leer parametros del terminal
import sys 

from scipy import signal
from math import log10
from scipy.ndimage import generic_laplace,uniform_filter,correlate,gaussian_filter
from enum import Enum
#-----------------------------------------



class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1

def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul

def fspecial(fltr,ws,**kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)


def _initial_check(GT,P):
#	assert GT.shape == P.shape, "Supplied images have different sizes " + \
#	str(GT.shape) + " and " + str(P.shape)
	
	#pongo esto, pues lo previo da error
	assert GT.shape == P.shape, "Supplied images have different sizes " + 	str(GT.shape) + " and " + str(P.shape)
	
	#añado Marisa
	if GT.shape != P.shape:
	   syst.exit()
	   
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT.astype(np.float64),P.astype(np.float64)


def _power_complex(a,b):
	return a.astype('complex') ** b



#HASTA AQUI CÓDIGO AUXILIAR PARA CALCULAR VARIOS DE LOS ERRORES




# Error cuadrático medio (MSE - MEAN SQUARED ERROR) 
def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""
	
	GT,P = _initial_check(GT,P)


	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)


# Error absoluto medio (MAE - MEAN ABSOLUTE ERROR)
def mae (GT,P):
	"""calculates mean absolute error (mae).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mae value.
	"""
	
	GT,P = _initial_check(GT,P)


#	return np.abs((GT.astype(np.float64)-P.astype(np.float64))**2)
	return np.mean(np.abs(GT.astype(np.float64)-P.astype(np.float64)))
	

# Error PSNR (Peak-signal-to-noise ratio)
def psnr (GT,P,MAX=None):
	"""calculates peak signal-to-noise ratio (psnr).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- psnr value in dB.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	mse_value = mse(GT,P)
 
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10(MAX**2 /mse_value)



# SSIM (Structural similarity index)
# Realmente calcula la media del error (MSSIM), pues hace la media del valor
# SSIM obtenido para los pixels de una ventana que se desliza sobre la imagen
# No confundir con el error MS-SSIM (la fórmula es diferente)
def _ssim_single (GT,P,ws,C1,C2,fltr_specs,mode):
	win = fspecial(**fltr_specs)

	GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)
	sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,win,mode,sums=(GT_sum_sq,P_sum_sq,GT_P_sum_mul))

	assert C1 > 0
	assert C2 > 0

	ssim_map = ((2*GT_P_sum_mul + C1)*(2*sigmaGT_P + C2))/((GT_sum_sq + P_sum_sq + C1)*(sigmaGT_sq + sigmaP_sq + C2))
	cs_map = (2*sigmaGT_P + C2)/(sigmaGT_sq + sigmaP_sq + C2)

	
	return np.mean(ssim_map), np.mean(cs_map)


def ssim (GT,P,ws=11,K1=0.01,K2=0.03,MAX=None,fltr_specs=None,mode='valid'):
	"""calculates structural similarity index (ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  tuple -- ssim value, cs value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	if fltr_specs is None:
		fltr_specs=dict(fltr=Filter.UNIFORM,ws=ws)

	C1 = (K1*MAX)**2
	C2 = (K2*MAX)**2

	ssims = []
	css = []
	for i in range(GT.shape[2]):
		ssim,cs = _ssim_single(GT[:,:,i],P[:,:,i],ws,C1,C2,fltr_specs,mode)
		ssims.append(ssim)
		css.append(cs)
	return np.mean(ssims),np.mean(css)


# Error MS-SSIM (multi-scale structural similarity index)
def msssim (GT,P,weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],ws=11,K1=0.01,K2=0.03,MAX=None):
	"""calculates multi-scale structural similarity index (ms-ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
	:param ws: sliding window size (default = 11).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- ms-ssim value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	scales = len(weights)

	fltr_specs = dict(fltr=Filter.GAUSSIAN,sigma=1.5,ws=11)

	if isinstance(weights, list):
		weights = np.array(weights)

	mssim = []
	mcs = []
	for _ in range(scales):
		_ssim, _cs = ssim(GT, P, ws=ws,K1=K1,K2=K2,MAX=MAX,fltr_specs=fltr_specs)
		mssim.append(_ssim)
		mcs.append(_cs)

		filtered = [uniform_filter(im, 2) for im in [GT, P]]
		GT, P = [x[::2, ::2, :] for x in filtered]

	mssim = np.array(mssim,dtype=np.float64)
	mcs = np.array(mcs,dtype=np.float64)

	return np.prod(_power_complex(mcs[:scales-1],weights[:scales-1])) * _power_complex(mssim[scales-1],weights[scales-1])





# PROGRAMA PRINCIPAL
# Intento leer del terminal los nombres de los dos ficheros de imagen
if len(sys.argv) == 3:

   # se copia el nombre de las dos imágenes
   figura1 = sys.argv[1]
   figura2 = sys.argv[2]

   #print("procesando las imagenes: |"+ figura1 + "| y |"+ figura2 + "|")
   #print(figura1 + " "+ figura2) #NOMBRES DE LOS DOS FICHEROS
   print(figura1, end=" " )
 
   # se leen ambas imágenes (son imágenes en color)
   FIG1 = cv2.imread(figura1,cv2.IMREAD_COLOR)  
   FIG2 = cv2.imread(figura2,cv2.IMREAD_COLOR)    

   #pongo aqui la comprobación de imagenes del mismo tamaño
   # que aparecía en casi todas las funcione de calculo de los errores.
   # Así puedo ejecutarla sólo una vez antes de calcular todos los errores
   _initial_check(FIG1, FIG2)
   

# Si no se indicaron los dos ficheros de imagen, se informa y concluye la ejecución   
else:
   print("Debe indicar dos ficheros de imagen")
   exit()


# Cada función comprueba el tamaño de las imágenes.
# Sería mejor hacer esta comprobación una sola vez antes de calcular
# varios errores, pero la he dejado por si sólo se quiere calcular uno
# o reutilizar el código.
   
# errores mostrados como salida del programa:
print("Errores calculados: MSE PSNR MAE SSIM MS-SSIM")


# ------------------- calculo el MSE -------------------
resul_mse = mse (FIG1, FIG2)

#MULTIPLICO POR 3, que es como lo hago en mis programas
mse_yo = resul_mse * 3


# ------------------- calculo el PSNR -------------------
# No es necesario llamar a la función que calcula el PSNR, pues
# acabo de calcular el MSE. La función considera el MSE que está
# multiplicado por 3 
if mse_yo == 0.:
   psnr_yo = np.inf  # para MSE 0, el PSNR sería infinito
else:
   psnr_yo = 10 * np.log10(255**2 /mse_yo)


# ------------------- calculo el MAE -------------------
#MULTIPLICO POR 3, que es como lo hago en mis programas
mae_yo = 3*mae(FIG1, FIG2)


# ------------------- calculo el error SSIM -------------------
resul_ssim, resul_cs =ssim(FIG1, FIG2) 


# ------------------- calculo el error MS-SSIM -------------------
resul_msssim = msssim(FIG1, FIG2)  


#escribo todos los errores en una sola linea para facilitar el procesamiento de múltiples imágenes
# .real en el último dato evita que aparezca con el formato (0.9786+0j)
print( mse_yo, psnr_yo, mae_yo, resul_ssim, resul_msssim.real)

