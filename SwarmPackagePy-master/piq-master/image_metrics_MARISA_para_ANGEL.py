import torch
import piq
from skimage.io import imread
import argparse
import os


@torch.no_grad()
def main():


    parser = argparse.ArgumentParser(description="Compute image quality metrics.")
    parser.add_argument("imagenOriginal", type=str, help="Nombre de la imagen original.")
    parser.add_argument("imagenCuantizada", type=str, help="Nombre de la imagen cuantizada.")
    args = parser.parse_args()

    # Construir las rutas completas de las imágenes
    # Asumimos que las imágenes están en directorios hermanos a `piq-master`
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)  # Subir un nivel a la carpeta padre

    path_original = os.path.join(parent_dir, 'imagenes', args.imagenOriginal)
    path_cuantizada = os.path.join(parent_dir, 'imagenesCuantizadas', args.imagenCuantizada)

    # Read RGB image and it's noisy version
#    x = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...] / 255.
 #   y = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...] / 255.

    x = torch.tensor(imread(path_original)).permute(2, 0, 1)[None, ...] / 255.
    y = torch.tensor(imread(path_cuantizada)).permute(2, 0, 1)[None, ...] / 255.


    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()
        y = y.cuda()





    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(x, y, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    print(f"{fsim_index.item():0.4f}", end=' ')



    # To compute VIF as a measure, use lower case function from the library:
    vif_index: torch.Tensor = piq.vif_p(x, y, data_range=1.)
    # In order to use VIF as a loss function, use corresponding PyTorch class:
    print(f"{vif_index.item():0.4f}", end= ' ')



if __name__ == '__main__':
    main()
