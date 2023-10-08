import os
import shutil
from PIL import Image

source_dir = '/home/jli/Cell_cycleGAN/results/staining/Fakulty_to_GOWT_grey/epoch15/test_latest/images'
dest_dir = '/home/jli/Cell_cycleGAN/stain_postprocess/original_data'

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop over all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file ends with '_fake.png'
    if filename.endswith('_fake.png'):
        # Construct the full path to the source file
        src_path = os.path.join(source_dir, filename)
        # Construct the new filename with the suffix 'stained.tif'
        new_filename = os.path.splitext(filename)[0] + '_stained.tif'
        # Construct the full path to the destination file
        dest_path = os.path.join(dest_dir, new_filename)
        # Convert the image to TIFF format and save it to the destination directory
        im = Image.open(src_path)
        im.save(dest_path, format='TIFF')