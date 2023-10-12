"""
This file is used to process the images to remove the bubbles and noise in the images.
These steps are aimed for stained cell images with blue color for interests parts and white color for background.
The steps are as follows:
1. Apply minimum filter with 7x7 kernel. to get a more general background and cell mask.
2. Create cell mask based on blue value. If the blue value is greater than red value and green value by 25, then it is a cell.
3. Remove small patches in the cell mask.
4. Dilate the cell mask to get a more general cell mask.
5. Invert the cell mask to get background mask.
6. Replace background areas with specific patches.
7. Gaussian blur.
8. Denoise.
"""

import cv2
import numpy as np
import random
import os

import random
import numpy as np

def replace_with_specific_patch(image, mask, background_patch):
    h_patch, w_patch, _ = background_patch.shape
    h_img, w_img, _ = image.shape
    patch_size = 5
    for row in range(0, h_img, patch_size):
        for col in range(0, w_img, patch_size):
            # Determine the end row and column indices for the slice
            end_row = min(row + patch_size, h_img)
            end_col = min(col + patch_size, w_img)

            # Extract sub-mask from the mask
            sub_mask = mask[row:end_row, col:end_col]

            # Check if all pixels in the patch are background
            if np.all(sub_mask == 255):
                # Choose a random 15x15 patch from the background patch
                x_bg = random.randint(0, h_patch - patch_size)
                y_bg = random.randint(0, w_patch - patch_size)
                sub_patch = background_patch[x_bg:x_bg + patch_size, y_bg:y_bg + patch_size]

                # Get the shape of the sub-patch to fit into the slice
                sub_shape = sub_mask.shape[:2]

                # Paste the random patch into the image
                image[row:end_row, col:end_col] = sub_patch[:sub_shape[0], :sub_shape[1]]

    return image



# Initial patch coordinates
x_coord, y_coord = 414, 816  # Replace these with your chosen coordinates
patch_size = 210

# Read the initial image and crop the patch
init_image = cv2.imread('../Test_Dataset/Origin/cell_00001.bmp')
background_patch = init_image[x_coord:x_coord + patch_size, y_coord:y_coord + patch_size]
# Save the background patch
cv2.imwrite('/home/frozen/Report_Cell_CycleGAN/background_patch.png', background_patch)

# Directories
input_dir = '/home/frozen/Report_Cell_CycleGAN/Origin/'
min_filter_dir = '/home/frozen/Report_Cell_CycleGAN/Min_Filter/'
blue_mask_dir = '/home/frozen/Report_Cell_CycleGAN/Blue_Mask/'
remove_patch_dir = '/home/frozen/Report_Cell_CycleGAN/Remove_Patch/'
dilate_dir = '/home/frozen/Report_Cell_CycleGAN/Mask_Dilation/'
background_mask_dir = '/home/frozen/Report_Cell_CycleGAN/Background_Mask/'
background_replacement_dir = '/home/frozen/Report_Cell_CycleGAN/Background_Replacement/'
output_dir = '/home/frozen/Report_Cell_CycleGAN/Final_Result/'
temp_dir = 'Test_Dataset/Temp/'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through BMP images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.bmp') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(input_dir, filename))

        # Apply minimum filter with 7x7 kernel
        erode_kernel = np.ones((7, 7), np.uint8)
         # Apply 7x7 minimum filter
        img_min_filter = cv2.erode(img, np.ones((7, 7), np.uint8))

        # Store the image with minimum filter applied
        cv2.imwrite(os.path.join(min_filter_dir, f"Min_Filter_{filename}"), img_min_filter)

         # Create cell mask based on blue value
        blue_channel = img_min_filter[:, :, 0]
        red_channel = img_min_filter[:, :, 2]
        green_channel = img_min_filter[:, :, 1]
        cell_mask = np.zeros_like(blue_channel)
        # TODO: FIX the data overflow issue
        cell_indices = np.where((blue_channel > red_channel + 25) & (blue_channel > green_channel + 25))
        cell_mask[cell_indices] = 255

        # Save cell mask as image
        cv2.imwrite(os.path.join(blue_mask_dir, f"Blue_Mask_{filename}"), cell_mask)
        
        
        
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 625:  # Area less than 8x8
                cv2.drawContours(cell_mask, [contour], 0, 0, -1)

        # Save removed patch image
        cv2.imwrite(os.path.join(remove_patch_dir, f"Remove_Patch_{filename}"), cell_mask)

        # Dilation and remove small patches
        kernel = np.ones((5,5), np.uint8)
        cell_mask = cv2.dilate(cell_mask, kernel, iterations=1)
        # Save dilated mask
        cv2.imwrite(os.path.join(dilate_dir, f"Dilate_{filename}"), cell_mask)

        # Save dilated mask
        cv2.imwrite(os.path.join(dilate_dir, f"Dilate_{filename}"), cell_mask)

        # Invert cell mask to get background mask
        bg_mask = cv2.bitwise_not(cell_mask)
        # Save background mask
        cv2.imwrite(os.path.join(background_mask_dir, f"Background_Mask_{filename}"), bg_mask)

        # Replace background areas with specific patches
        img_processed = replace_with_specific_patch(img, bg_mask, background_patch)

        # Save background replaced image
        cv2.imwrite(os.path.join(background_replacement_dir, f"Background_Replacement_{filename}"), img_processed)

        # Gaussian blur
        img_processed = cv2.GaussianBlur(img_processed, (5, 5), 0)
        
        # Denoise
        img_processed = cv2.fastNlMeansDenoisingColored(img_processed, None, 10, 10, 7, 21)

        # Save processed image
        cv2.imwrite(os.path.join(output_dir, f"Processed_{filename}"), img_processed)
