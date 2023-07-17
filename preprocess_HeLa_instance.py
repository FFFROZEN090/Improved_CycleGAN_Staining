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
ROOT_DIR = 'Grund_Truth_Instance' #Zielordner
"""
color_white = (255,255,255)
color_black = (0,0,0)
color_red = (255,0,0)
color_green = (0,255,0)
color_blue = (0,0,255)
color_violet =(138,43,226)
color_orange = (255,127,36)
color_pink = (255,20,147)
color_yellow =(255,255,0)
color_dark_green =(0,139,0)
color_khaki =(238,230,133)
color_olive = (179,238,58)
color_cyan = (0,255,255)
"""
color_list = ((0,0,0),
            (244, 35, 232),
            (102, 102, 156),
            (190, 153, 153),
            (255,255,0),
            (0,255,255),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (170, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0,255,0),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
            (255,127,36),
            (0,139,0),
            )  
"""       
color_map = np.array([
        [0, 0, 0],
        [244, 35, 232],
        [102, 102, 156],
        [190, 153, 153],
        [255,255,0],
        [0,255,255],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [170, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0,255,0],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [255,127,36],
        [0,139,0]
    ], dtype=np.uint8)

"""
# Create new data dir    
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val'), exist_ok=True)

#Anzahl files im Ordner
count_files = len(root_studies)

#Beide Ordner durchgehen
for i in range(0,count_files):

      #Liste der Bilder im Ordner erstellen
      images = sorted(os.listdir(os.path.join(RAW_DATA_DIRECTORY, root_studies[i] + '_ST', 'SEG')))

      #Anzahl der Bilder speichern
      num_images = len(images)

      #Alle Bilder pro Ordner durchgehen
      for j in range(0,num_images):

            #Bild öffnen
            image = Image.open(os.path.join(RAW_DATA_DIRECTORY, root_studies[i]+ '_ST', 'SEG', images[j]))  

            #Bild in numpy array umwandeln
            img = np.array(image)

            #Bildmatrix transposen, da das Bild sonst auf den Kopf steht
            img = np.transpose(img) 

            #ermittle die Größe des Bildes, Ausgabe ist ein Tupel(512,512)
            shape_of_image = np.shape(img) 
            #speichere Größe als einzelner Wert für Schleifen
            shape_of_image = shape_of_image[0] 

            unique_values_of_image = np.unique(img) #Unterschiedliche Werte in Array ermitteln
            #print(unique_values_of_image)
            anzahl_unique_values = len(unique_values_of_image) #Anzahl der Unterschiedlichen Werte speichern
            
            #norm the indizes to 0-19 so I dont have arbitrary index like 0,3,15,34 etc
            for d in range(0,shape_of_image):
                  for e in range(0,shape_of_image):
                        for f in range(0,anzahl_unique_values):
                              if img[d][e] == unique_values_of_image[f]:
                                    unique_values_of_image[unique_values_of_image == f] = f
                                    img[d][e] = f
            unique_values_of_image = np.unique(img)
            print(unique_values_of_image)
            print(np.unique(img))
            
            #numpy array wieder in Bild umwandeln
            image = Image.fromarray(img)

            #Bild in rgb Modus umwandeln, um rgb Werte ändern zu können
            rgb_im = image.convert("RGB")
            
            #alles Pixel im Bild durchgehen und Wert ändern
            for a in range(0,shape_of_image):
                  for b in range(0,shape_of_image):
                        for c in range(0,anzahl_unique_values):
                              if img[a][b] == unique_values_of_image[c]:
                                    rgb_im.putpixel((a, b), color_list[c])
            
            
            print(j)
            image_name_new = root_studies[i] + '_' + str(j) + '.tiff'
            if j % 5 !=0:
                  rgb_im.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train', image_name_new), exist_ok=True)
            else:
                  rgb_im.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val', image_name_new), exist_ok=True)
            image.close()
            rgb_im.close()

"""
for i in range(0,shape_of_image):
    for j in range(0,shape_of_image):
            if img[i][j] >= 1:
                  img[i][j]=1

"""