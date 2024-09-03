#!/bin/bash
# ---------------------------------------------------------------------
# Script para aplicar el método PSO a imágenes PPM del conjunto CQ-100
# y calcular después varias medidas de error.
#
# Se supone que el programa que aplica un método de enjambre genera una imagen cuantizada, 
# con un formato predefinido para su nombre.
# Por ejemplo, el formato del nombre de fichero que yo creo para mi imagen cuantizada desde el código python,
# para el método PSO, reduciendo a 32 colores y aplicado a la primera imagen del listado, es:
#    PSO_32_adirondack_chairs.ppm
#  (la extensión del nombre del fichero es la misma que tenga el fichero original.
#   Es decir, si leo una imagen ppm, escribo una ppm, si leo tif escribo tif, ...)
# ---------------------------------------------------------------------

# Acrónimo del algoritmo de enjambre que voy a aplicar 
# PSO -> PSO
# FA -> Luciernagas
# BA -> ballenas
# GWO -> Lobos
# ABA -> Abejas
#"FA" "BA" "GWO" "ABA"
iteraciones=15
individuos=20

for ALGO in "PSO" "FA" "BA" "GWO" "ABA"
do


# Bucle para recorrer todas las funciones que acepta el programa
# MSE, MAE, SSIM, MSSIM
for FUNC in "MSE" "MAE" "SSIM" "MSSIM" 
do


#ESCRIBO RÓTULOS UNA SOLA VEZ EN LOS FICHEROS QUE ALMACENARÁN LOS ERRORES.
# Separo los resultados de cada tamaño de paleta en un fichero diferente
for C in 32 64 128 256; # para cada tamaño de paleta cuantizada
do 
echo "MSE PSNR MAE SSIM MS-SSIM" >> IQI_${ALGO}_${FUNC}_${C}.txt
 


# Bucle que recorre desde 1 hasta el número de iteraciones y los imprime en una sola línea para formar la cabecera 
for (( i=1; i<=iteraciones; i++ ))
do
    echo -n "$i " >> salida_${ALGO}_${FUNC}_${C}.txt #lo escribo en mi txt 
done
echo "Tiempo" >> salida_${ALGO}_${FUNC}_${C}.txt
echo >> salida_${ALGO}_${FUNC}_${C}.txt # Añade un salto de linea

# Para cada imagen del conjunto de 100 CQ-100 
#for F in adirondack_chairs.ppm astro_bodies.ppm astronaut.ppm balinese_dancer.ppm ball_caps.ppm birthday_baloons.ppm bosnian_pine_needle.ppm buggy.ppm calaveras.ppm carrots.ppm chalk_pastels.ppm chicken_dish.ppm chili_peppers.ppm clownfish.ppm color_chart.ppm  color_checker.ppm coloring_pencils.ppm columbia_crew.ppm common_jezebel.ppm common_lantanas.ppm  cosmic_vista.ppm  craft_cards.ppm crepe_paper.ppm cruise_ship.ppm curler.ppm daisy_bouquet.ppm daisy_poster.ppm  easter_egg_basket.ppm  easter_eggs.ppm eastern_rosella.ppm felt_ball_trivet.ppm fishing_nets.ppm floating_market.ppm fruit_dessert.ppm fruit_stand.ppm  fruits.ppm fruits_veggies.ppm german_hot_air_balloon.ppm girl.ppm gourds.ppm grilled_food.ppm  hard_candy.ppm italian_hot_air_balloon.ppm jacksons_chameleon.ppm king_penguin.ppm king_vulture.ppm  kingfisher.ppm korean_dancer.ppm lights.ppm macarons.ppm;
# macaws.ppm malayan_banded_pitta.ppm  mandarin_ducks.ppm mandarinfish.ppm mangoes.ppm marrakech_museum.ppm maya_beach.ppm  medicine_packets.ppm moroccan_babouches.ppm motocross.ppm motorcycle.ppm mural.ppm  nylon_cords.ppm paper_clips.ppm peacock.ppm pencils.ppm pigments.ppm  pink_mosque.ppm plushies.ppm prickly_pears.ppm puffin.ppm race_car.ppm  red_eyed_tree_frog.ppm red_knobbed_starfish.ppm rescue_helicopter.ppm rose_bouquet.ppm sagami_temple.ppm salad_bowl.ppm schoolgirls.ppm seattle_great_wheel.ppm shawls.ppm shopping_bags.ppm  siberian_tiger.ppm skiers.ppm spices.ppm sports_bicycles.ppm sun_parakeet.ppm tablet.ppm  textile_market.ppm trade_fair_tower.ppm traffic.ppm tulip_field.ppm umbrellas.ppm  veggie_pizza.ppm veggies.ppm venetian_lagoon.ppm vintage_cars.ppm wooden_toys.ppm  wool_carder_bee.ppm yasaka_pagoda.ppm ;
#do 

for F in adirondack_chairs.ppm
do

# mensaje para el terminal (método que voy a probar)
echo "algoritmo->$ALGO Cuantizada->$C Funcion-> $FUNC Imagen -> $F"



# Los errores calculados para sucesivas pruebas se guardarán en un fichero TXT cuyo nombre es de
# la forma:
#   IQI_PSO_32.txt   -> errores para el método PSO que usa una paleta de 32 colores
#   IQI_PSO_64.txt   -> errores para el método PSO que usa una paleta de 64 colores
#   IQI_PSO_128.txt  -> errores para el método PSO que usa una paleta de 128 colores
#   IQI_PSO_256.txt  -> errores para el método PSO que usa una paleta de 256 colores

#Escribo el nombre de la imagen original como rótulo del bloque de pruebas consecutivas
echo  -n "$F" >> IQI_${ALGO}_${FUNC}_${C}.txt


# NÚMERO DE TEST INDEPENDIENTES EJECUTADOS PARA UNA MISMA CONFIGURACIÓN (5, DE MOMENTO) POR AHORA LO QUITO PARA QUE SOLO SE LLAME UNA VEZ
#for TEST_INDEPEN in {1..5} 
#do 
 
#Antes de ejecutar el programa capturo el tiempo del inicio de la ejecucion para luego saber lo que tardo en ejecutarse el algoritmo
tiempoIni=$(date +%s)


# Ejecuto el código python que aplica un algoritmo de enjambre.
# Supongamos que se llama probar_PSO.py y necesita como parámetros:
# -- imagen a procesar
# -- numero de colores
# -- algoritmo a realizar
# -- funcion a procesar
# Como resultado de las operaciones, escribe en disco la imagen cuantizada con el nombre
# que se ajuste al forma indicado anteriormente.
# Los resultados de la ejecución del programa se vuelcan en un fichero con un
# nombre de la forma:
#   salida_PSO_32.txt   salida_PSO_64.txt   salida_PSO_128.txt   o salida_PSO_256.txt , en este ejemplo
python3 ejecutor.py ${F} ${C} ${ALGO} ${FUNC} ${iteraciones} ${individuos} >> salida_${ALGO}_${FUNC}_${C}.txt

# Capturar el tiempo de fin
tiempoFin=$(date +%s)
# Calcular la duración en segundos
duration=$(($tiempoFin - $tiempoIni))

#Escribo en el txt correspondiente la duracion de la ejecucion en segundos
echo $duration >> salida_${ALGO}_${FUNC}_${C}.txt 
#y un salto de linea para la siguiente ejecucion
echo >> salida_${ALGO}_${FUNC}_${C}.txt


# Calculo múltiples medidas de error sobre la imagen cuantizada que acabo de generar
python3 errores_cq.py ${F} ${ALGO}_${C}_${F} >> IQI_${ALGO}_${FUNC}_${C}.txt


# NOTA: aquí tendrías que ver la información que se vuelca a los TXT desde cada programa,
# para valorar si es necesario escribir información extra en cada linea de salida:
# nombre de la imagen que se procesa, tamaño de la población, número de iteraciones...
# para que sea facil entender con que valores de los parámetros se ha obtenido cada linea de resultados.
# Otra opción es que aparezca la información como rótulo de un bloque de lineas,
# pero lo verás mejor cuando hagas alguna prueba.

# Borro la imagen cuantizada, que ya no necesito (guardarlas todas me ocupa
# mucho disco duro)  
cd images
rm ${ALGO}_${C}_${F}
cd ..
  
#done #tests sucesivos
done  #imagen original
done  # colores de la paleta cuantizada
done # Funcion
done # ALgoritmo




