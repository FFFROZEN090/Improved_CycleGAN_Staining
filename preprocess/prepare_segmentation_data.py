import glob
import shutil
from PIL import Image
from util.util import assure_path_exists


dir = '/home/mr38/sds_hd/sd18a006/Marlen/datasets/segmentation/urothel/classes_12/TCGA-C4-A0F0-01Z-00-DX1.8EBBC2EF-DA6F-4901-838D-C1AC80E83E92/512/'
input_dir = dir + '*-labels.png'
output_images_dir = dir + 'trainA/'
output_labels_dir = dir + 'trainB/'
assure_path_exists(output_images_dir)
assure_path_exists(output_labels_dir)

if __name__ == '__main__':
    for label_file in glob.glob(input_dir):
         try:
           label = Image.open(label_file) # open the label file
           image_file = label_file[:-11] + ".png"
           image = Image.open(image_file) # open the image file
           label.verify() # verify that it is, in fact an image
           image.verify() # verify that it is, in fact an image
           shutil.move(image_file, output_images_dir)
           shutil.move(label_file, output_labels_dir)
         except (IOError, SyntaxError) as e:
           print('Bad file:', label_file) # print out the names of corrupt files

