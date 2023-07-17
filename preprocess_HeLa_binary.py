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
ROOT_DIR = 'Ground_Truth_Binary' #Zielordner

# Create new data dir    
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val'), exist_ok=True)

#Anzahl files im Ordner
count_files = len(root_studies)

color_white = (255,255,255)
color_black= (0,0,0)

#Beide Ordner durchgehen
for i in range(0,count_files):

      #Liste der Bilder im Ordner erstellen
      images = sorted(os.listdir(os.path.join(RAW_DATA_DIRECTORY, root_studies[i] + '_ST', 'SEG')))

      #Anzahl der Bilder speichern
      num_images = len(images)
      

      #Alle Bilder pro Ordner durchgehen
      #for j in range(0,num_images):
      for j in range(0,num_images):

            #Bild öffnen
            image = Image.open(os.path.join(RAW_DATA_DIRECTORY, root_studies[i]+ '_ST', 'SEG', images[j]))  

            #Bild in numpy array umwandeln
            img = np.array(image) 
            #print(img)

            unique_values_of_image = np.unique(img) #Unterschiedliche Werte in Array ermitteln
            
            anzahl_unique_values = len(unique_values_of_image) #Anzahl der Unterschiedlichen Werte speichern

            #ermittle die Größe des Bildes, Ausgabe ist ein Tupel(512,512)
            shape_of_image = np.shape(img) 
            #speichere Größe als einzelner Wert für Schleifen
            shape_of_image = shape_of_image[0] 
            img = np.transpose(img)

            #norm the indizes to 0-19 so I dont have arbitrary index like 0,3,15,34 etc
            for d in range(0,shape_of_image):
                  for e in range(0,shape_of_image):
                        for f in range(0,anzahl_unique_values):
                              if img[d][e] == unique_values_of_image[f]:
                                    unique_values_of_image[unique_values_of_image == f] = f
                                    img[d][e] = f
            unique_values_of_image = np.unique(img)
            # stain all cells white, and background black
            for c in range(0,shape_of_image):
                  for d in range(0,shape_of_image):
                        if img[c][d] !=0:
                               img[c][d] = 1
                        else:
                               img[c][d] = 0
                               
           
           
            #numpy array wieder in Bild umwandeln
            image = Image.fromarray(img)

            #Bild in rgb Modus umwandeln, um rgb Werte ändern zu können
            rgb_im = image.convert("RGB")
            
            
            
            
            #alles Pixel im Bild durchgehen und Wert ändern
            for a in range(0,shape_of_image):
                  for b in range(0,shape_of_image):
                        #r, g, b = rgb_im.getpixel((a, b))
                        color_white = (255,255,255)
                        if img[a][b]!=0:
                              
                              rgb_im.putpixel((a, b), color_white)
                        else:
                              rgb_im.putpixel((a, b), color_black)
                        
                        r, g, b = rgb_im.getpixel((a, b))
                        if r!=0 and r!=255:
                              print(r,g,b)
                              
                        #if r>=1:
                              #rgb_im.putpixel((a, b), color_white)
                        #else:
                              #rgb_im.putpixel((a, b), color_black) 
            
            
            print(j)
            image_name_new = root_studies[i] + '_' + str(j) + '.tiff'
            if j % 5 !=0:
                  rgb_im.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'train', image_name_new), exist_ok=True)
            else:
                  rgb_im.save(os.path.join(RAW_DATA_DIRECTORY, ROOT_DIR, 'val', image_name_new), exist_ok=True)
            image.close()
            rgb_im.close()

           