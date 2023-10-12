import os
import cv2

def invert_images(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        # Check if the file is a TIFF image
        if filename.endswith('.tiff'):
            filepath = os.path.join(input_directory, filename)
            
            # Read the image in grayscale
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Invert the image
            inverted_image = 255 - image
            
            # Create output filepath
            output_filepath = os.path.join(output_directory, f"inverted_{filename}")
            
            # Save the inverted image
            cv2.imwrite(output_filepath, inverted_image)
            print(f"Inverted and saved: {filename}")

# Example usage
input_directory_path = '/home/frozen/Data_Sets/train_GOWT_binary'  # Replace with your input directory path
output_directory_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/AA/testA'  # Replace with your output directory path
invert_images(input_directory_path, output_directory_path)
