import cv2
import os

def downsample_and_tile(input_path, output_path):
    # Create output directory if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Loop through each image in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.bmp'):
            img_path = os.path.join(input_path, filename)
            
            # Read image
            img = cv2.imread(img_path)
            
            # Resize to 1024x1024
            resized_img = cv2.resize(img, (1024, 1024))
            
            # Split into 4 512x512 sub-images
            sub_image1 = resized_img[0:512, 0:512]
            sub_image2 = resized_img[0:512, 512:1024]
            sub_image3 = resized_img[512:1024, 0:512]
            sub_image4 = resized_img[512:1024, 512:1024]
            
            # Save sub-images
            cv2.imwrite(os.path.join(output_path, f"{filename.split('.')[0]}_1.bmp"), sub_image1)
            cv2.imwrite(os.path.join(output_path, f"{filename.split('.')[0]}_2.bmp"), sub_image2)
            cv2.imwrite(os.path.join(output_path, f"{filename.split('.')[0]}_3.bmp"), sub_image3)
            cv2.imwrite(os.path.join(output_path, f"{filename.split('.')[0]}_4.bmp"), sub_image4)

def upsample_and_tile(input_path, output_path):
    # Create output directory if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loop through each image in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.bmp'):
            img_path = os.path.join(input_path, filename)

            # Read image
            img = cv2.imread(img_path)

            # Resize to 2048x2048
            resized_img = cv2.resize(img, (2048, 2048))

            # Split into 4 1024x1024 sub-images
            idx = 0
            for i in range(0, 2048, 1024):
                for j in range(0, 2048, 1024):
                    sub_image = resized_img[i:i+1024, j:j+1024]
                    
                    # Save sub-images
                    cv2.imwrite(os.path.join(output_path, f"{filename.split('.')[0]}_{idx}.bmp"), sub_image)
                    idx += 1

# Example usage
input_directory = 'Test_Dataset/Processed_Outliers_Free'  # Replace with your input directory
output_directory = 'Test_Dataset/Processed_Upsample_Tiled_OutliersFree'  # Replace with your output directory
upsample_and_tile(input_directory, output_directory)