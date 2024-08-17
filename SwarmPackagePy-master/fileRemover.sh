for ALGO in "PSO" "FA" "BA" "GWO" "ABA"
do

for C in 32 64 128 256 ; 
do 

rm salida_${ALGO}_${C}.txt
rm IQI_${ALGO}_${C}.txt

done 
done