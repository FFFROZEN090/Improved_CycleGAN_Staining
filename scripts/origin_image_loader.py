import cv2
import os

# Define directories
src_dir = "/home/frozen/Experiments_Repitition/Cell_cycleGAN/AA/testA"
dest_dir = "/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/Origin"

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.tiff'):
        # Create full file path
        img_path = os.path.join(src_dir, filename)
        
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Create new filename
        new_filename = filename.replace('inverted_', '').replace('.tiff', '.png')
        
        # Save as PNG in destination directory
        dest_path = os.path.join(dest_dir, new_filename)
        cv2.imwrite(dest_path, img)