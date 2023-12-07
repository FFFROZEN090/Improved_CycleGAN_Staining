import os
from PIL import Image

# Set the directory where your BMP images are stored
dir_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Training_Datasets/Blue_Bubble_tiledGOWT_Inverse/trainB'

# Function to flip and rotate images
def process_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.bmp'):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)

            # Create a dictionary to hold original and flipped images
            images_to_rotate = {
                'original': image,
                'flipped_x': image.transpose(Image.FLIP_LEFT_RIGHT),
                'flipped_y': image.transpose(Image.FLIP_TOP_BOTTOM)
            }

            # Save the flipped images
            for key, img in images_to_rotate.items():
                if 'flipped' in key:  # Save the flipped images with a prefix
                    img.save(os.path.join(directory, f'{key}_{filename}'))

            # Now rotate all images (original and flipped)
            for key, img in images_to_rotate.items():
                for angle in [90, 180, 270]:
                    rotated = img.rotate(angle)
                    rotated.save(os.path.join(directory, f'{key}_rotated_{angle}_{filename}'))

# Call the function with your directory
process_images(dir_path)
