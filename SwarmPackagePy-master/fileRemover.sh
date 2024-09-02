for ALGO in "PSO" "FA" "BA" "GWO" "ABA"
do

for FUNC in "MSE" "MAE" "SSIM" "MSSIM" 
do
for C in 32 64 128 256 ; 
do 

rm salida_${ALGO}_${FUNC}_${C}.txt
rm IQI_${ALGO}_${FUNC}_${C}.txt

done 
done
done
