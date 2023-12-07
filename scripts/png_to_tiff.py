from PIL import Image
import os

def convert_png_to_tiff(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(directory, filename))
            tiff_filename = os.path.splitext(filename)[0] + '.tiff'
            img.save(os.path.join(directory, tiff_filename))

# Specify the directory containing the .png images
directory = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/training_dataset_tiledGOWT_Fakulty_Inverse_valA/Epoch5'
convert_png_to_tiff(directory)