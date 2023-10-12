import cv2
import os
import csv
import numpy as np

import cv2
import os
import csv
import numpy as np

def calculate_RGB_stats(image, lower_bound, upper_bound):
    stats = []
    for i, color in enumerate(['b', 'g', 'r']):
        single_channel = image[:,:,i]
        lower = lower_bound[i]
        upper = upper_bound[i]
        mask = cv2.inRange(single_channel, lower, upper)
        num_pixels = cv2.countNonZero(mask)
        stats.append(num_pixels)
    return stats

# Path to directory containing the bmp images
directory = 'Test_Dataset/Origin/'

# Initialize CSV file
with open("image_RGB_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "R-5_to_R+5", "G-5_to_G+5", "B-5_to_B+5"])

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.bmp'):
            filepath = os.path.join(directory, filename)
            
            # Read the image
            img = cv2.imread(filepath)

            # Calculate the mean value of each channel
            mean_val = np.mean(img, axis=(0, 1))
            mean_R, mean_G, mean_B = mean_val[2], mean_val[1], mean_val[0]  # OpenCV uses BGR format

            # Define the lower and upper bounds for R, G, B
            lower_bound = np.array([mean_B - 5, mean_G - 5, mean_R - 5], dtype="uint8")
            upper_bound = np.array([mean_B + 5, mean_G + 5, mean_R + 5], dtype="uint8")

            # Get the statistics for the specified range
            stats = calculate_RGB_stats(img, lower_bound, upper_bound)
            
            # Save the statistics to CSV file
            writer.writerow([filename] + stats)

print("Statistics saved to image_RGB_stats.csv")

