import os
import sys
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time
import cv2


#DATA_DIRECTORY = './HeLa_tif'
RAW_DATA_DIRECTORY = './DIC-C2DH-HeLa' #Ort der HeLa raw data
root_studies = '01', '02'
#root_studies = 'mask_train', 'mask_val'
ROOT_DIR = 'original_HeLa' #Zielordner

# Create new data dir    
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val'), exist_ok=True)

#Anzahl files im Ordner
count_files = len(root_studies)

#Beide Ordner durchgehen
for i in range(0,count_files):

      #Liste der Bilder im Ordner erstellen
      images = sorted(os.listdir(os.path.join(RAW_DATA_DIRECTORY, root_studies[i])))

      #Anzahl der Bilder speichern
      num_images = len(images)

      #Alle Bilder pro Ordner durchgehen
      for j in range(0,num_images):
            #Bild öffnen
            image = Image.open(os.path.join(RAW_DATA_DIRECTORY, root_studies[i], images[j]))

            print(j)
            image_name_new = root_studies[i] + '_' + str(j) + '.tiff'
            if j % 5 !=0:
                  image.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train', image_name_new), exist_ok=True)
            else:
                  image.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val', image_name_new), exist_ok=True)
            image.close()
            

