import os
import shutil

def copy_tiff_files(src_directory, dst_directory):
    for filename in os.listdir(src_directory):
        if filename.endswith('.tiff'):
            src_filepath = os.path.join(src_directory, filename)
            dst_filepath = os.path.join(dst_directory, filename)
            shutil.copy(src_filepath, dst_filepath)

# Specify the source and destination directories
src_directory = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/training_dataset_tiledGOWT_Fakulty_Inverse/Epoch5'
dst_directory = '/home/frozen/Experiments_Repitition/Cell_GAN/GOWT_Inverse_Stained_Epoch5/Input/train'

# Create destination directory if it doesn't exist
os.makedirs(dst_directory, exist_ok=True)

copy_tiff_files(src_directory, dst_directory)