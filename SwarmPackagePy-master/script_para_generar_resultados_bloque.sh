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
ALGO="PSO"


# Los errores calculados para sucesivas pruebas se guardarán en un fichero TXT cuyo nombre es de
# la forma:
#   IQI_PSO_32.txt   -> errores para el método PSO que usa una paleta de 32 colores
#   IQI_PSO_64.txt   -> errores para el método PSO que usa una paleta de 64 colores
#   IQI_PSO_128.txt  -> errores para el método PSO que usa una paleta de 128 colores
#   IQI_PSO_256.txt  -> errores para el método PSO que usa una paleta de 256 colores

# mensaje para el terminal (método que voy a probar)
echo "algoritmo->$ALGO"  



#ESCRIBO RÓTULOS UNA SOLA VEZ EN LOS FICHEROS QUE ALMACENARÁN LOS ERRORES.
# Separo los resultados de cada tamaño de paleta en un fichero diferente
for C in 32 64 128 256; # para cada tamaño de paleta cuantizada
do 
echo "MSE PSNR MAE SSIM MS-SSIM" >> IQI_${ALGO}_${C}.txt
done



# Para cada imagen del conjunto de 100 CQ-100 
#for F in adirondack_chairs.ppm astro_bodies.ppm astronaut.ppm balinese_dancer.ppm ball_caps.ppm birthday_baloons.ppm bosnian_pine_needle.ppm buggy.ppm calaveras.ppm carrots.ppm chalk_pastels.ppm chicken_dish.ppm chili_peppers.ppm clownfish.ppm color_chart.ppm  color_checker.ppm coloring_pencils.ppm columbia_crew.ppm common_jezebel.ppm common_lantanas.ppm  cosmic_vista.ppm  craft_cards.ppm crepe_paper.ppm cruise_ship.ppm curler.ppm daisy_bouquet.ppm daisy_poster.ppm  easter_egg_basket.ppm  easter_eggs.ppm eastern_rosella.ppm felt_ball_trivet.ppm fishing_nets.ppm floating_market.ppm fruit_dessert.ppm fruit_stand.ppm  fruits.ppm fruits_veggies.ppm german_hot_air_balloon.ppm girl.ppm gourds.ppm grilled_food.ppm  hard_candy.ppm italian_hot_air_balloon.ppm jacksons_chameleon.ppm king_penguin.ppm king_vulture.ppm  kingfisher.ppm korean_dancer.ppm lights.ppm macarons.ppm macaws.ppm malayan_banded_pitta.ppm  mandarin_ducks.ppm mandarinfish.ppm mangoes.ppm marrakech_museum.ppm maya_beach.ppm  medicine_packets.ppm moroccan_babouches.ppm motocross.ppm motorcycle.ppm mural.ppm  nylon_cords.ppm paper_clips.ppm peacock.ppm pencils.ppm pigments.ppm  pink_mosque.ppm plushies.ppm prickly_pears.ppm puffin.ppm race_car.ppm  red_eyed_tree_frog.ppm red_knobbed_starfish.ppm rescue_helicopter.ppm rose_bouquet.ppm sagami_temple.ppm salad_bowl.ppm schoolgirls.ppm seattle_great_wheel.ppm shawls.ppm shopping_bags.ppm  siberian_tiger.ppm skiers.ppm spices.ppm sports_bicycles.ppm sun_parakeet.ppm tablet.ppm  textile_market.ppm trade_fair_tower.ppm traffic.ppm tulip_field.ppm umbrellas.ppm  veggie_pizza.ppm veggies.ppm venetian_lagoon.ppm vintage_cars.ppm wooden_toys.ppm  wool_carder_bee.ppm yasaka_pagoda.ppm ;
 #do 
for F in mandril.tif
 do

#Escribo el nombre de la imagen original como rótulo del bloque de pruebas consecutivas
echo  -n "$F" >> IQI_${ALGO}_${C}.txt



# Para cada tamaño de paleta cuantizada
for C in 32 64 128 256 ; 
do 

# NÚMERO DE TEST INDEPENDIENTES EJECUTADOS PARA UNA MISMA CONFIGURACIÓN (5, DE MOMENTO)
for TEST_INDEPEN in {1. .5} 
do 
 
# Ejecuto el código python que aplica un algoritmo de enjambre.
# Supongamos que se llama probar_PSO.py y necesita como parámetros:
# -- imagen a procesar
# -- tamaño de la poblacion
# Como resultado de las operaciones, escribe en disco la imagen cuantizada con el nombre
# que se ajuste al forma indicado anteriormente.
# Los resultados de la ejecución del programa se vuelcan en un fichero con un
# nombre de la forma:
#   salida_PSO_32.txt   salida_PSO_64.txt   salida_PSO_128.txt   o salida_PSO_256.txt , en este ejemplo
python3 ejecutor.py ${F} 10 >> salida_${ALGO}_${C}.txt


# Calculo múltiples medidas de error sobre la imagen cuantizada que acabo de generar
python3 errores_cq.py ${F} ${ALGO}_${C}_${F} >> IQI_${ALGO}_${C}.txt


# NOTA: aquí tendrías que ver la información que se vuelca a los TXT desde cada programa,
# para valorar si es necesario escribir información extra en cada linea de salida:
# nombre de la imagen que se procesa, tamaño de la población, número de iteraciones...
# para que sea facil entender con que valores de los parámetros se ha obtenido cada linea de resultados.
# Otra opción es que aparezca la información como rótulo de un bloque de lineas,
# pero lo verás mejor cuando hagas alguna prueba.

# Borro la imagen cuantizada, que ya no necesito (guardarlas todas me ocupa
# mucho disco duro)  
rm ${ALGO}_${C}_${F}
  
done #tests sucesivos
done  # colores de la paleta cuantizada
done  #imagen original



