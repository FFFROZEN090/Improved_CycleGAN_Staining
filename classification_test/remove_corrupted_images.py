import glob
import shutil
from PIL import Image
from classification_test.util import assure_path_exists

dir = '/home/cw9/sds_hd/sd18a006/Marlen/datasets/stainNormalization/stainGAN_camelyon16'
input_dir = dir + '/**/*.png'
output_images_dir = dir + '/corrupted/'

if __name__ == '__main__':
    assure_path_exists(output_images_dir)

    for image_file in glob.glob(input_dir, recursive=True):
         try:
           image = Image.open(image_file) # open the image file
           image.verify() # verify that it is, in fact an image
         except (IOError, SyntaxError) as e:
           print('Bad file:', image_file) # print out the names of corrupt files
           image.close()
           shutil.move(image_file, output_images_dir)

