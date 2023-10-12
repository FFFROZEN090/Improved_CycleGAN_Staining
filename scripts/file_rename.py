import os
import cv2

def rename_tif_files(directory):
    i = 1  # Counter to prepend to filename
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            new_filename = f"02_{filename}"
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

            i += 1  # Increment counter



def flip_images(directory):
    for filename in os.listdir(directory):
        # Check if the file is an image (You can add more extensions here)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.tif'):
            filepath = os.path.join(directory, filename)
            
            # Read the image
            image = cv2.imread(filepath)
            
            # Flip the image along the y-axis
            flipped_image = cv2.flip(image, 1)
            
            # Create a new filename
            new_filename = "flipped_y_" + filename
            new_filepath = os.path.join(directory, new_filename)
            
            # Save the flipped image
            cv2.imwrite(new_filepath, flipped_image)
            print(f"Flipped and saved: {new_filename}")

def rotate_images(directory):
    for filename in os.listdir(directory):
        # Check if the file is an image (You can add more extensions here)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.tif'):
            filepath = os.path.join(directory, filename)
            
            # Read the image
            image = cv2.imread(filepath)
            
            # Rotate the image by 90, 180, and 270 degrees
            for angle in [90, 180, 270]:
                # Using cv2.rotate() method
                if angle == 90:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Create a new filename
                new_filename = f"rotated_{angle}_" + filename
                new_filepath = os.path.join(directory, new_filename)
                
                # Save the rotated image
                cv2.imwrite(new_filepath, rotated_image)
                print(f"Rotated and saved: {new_filename}")

def downsample_images(directory):
    for filename in os.listdir(directory):
        # Check if the file is an image (You can add more extensions here)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.tif'):
            filepath = os.path.join(directory, filename)
            
            # Read the image
            image = cv2.imread(filepath)
            
            # Resize the image to 1024x1024
            resized_image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            
            # Create a new filename
            new_filename = f"down_" + filename
            new_filepath = os.path.join(directory, new_filename)
            
            # Save the resized image
            cv2.imwrite(new_filepath, resized_image)
            print(f"Resized and saved: {new_filename}")

# Example usage
directory_path = 'training_dataset/testB'  # Replace with your directory path
rotate_images(directory_path)
