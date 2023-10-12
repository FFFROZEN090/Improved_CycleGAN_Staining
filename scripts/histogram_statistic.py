import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_histogram(image):
    hist_R = np.zeros((256,))
    hist_G = np.zeros((256,))
    hist_B = np.zeros((256,))

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        if i == 0:
            hist_B += hist.ravel()
        elif i == 1:
            hist_G += hist.ravel()
        elif i == 2:
            hist_R += hist.ravel()
    
    return hist_R, hist_G, hist_B

directory_path = 'Test_Dataset/Origin/'  # Replace with the path of your images directory

aggregated_hist_R = np.zeros((256,))
aggregated_hist_G = np.zeros((256,))
aggregated_hist_B = np.zeros((256,))

for filename in os.listdir(directory_path):
    if filename.endswith(('.jpg', '.png', '.bmp')):
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)
        hist_R, hist_G, hist_B = calculate_histogram(image)
        aggregated_hist_R += hist_R
        aggregated_hist_G += hist_G
        aggregated_hist_B += hist_B

plt.figure()
plt.title("Aggregated RGB Histogram (Bar Chart)")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# Create an index for each tick position
ind = np.arange(256)

# 0.3 is the width of the bars in the bar chart
plt.bar(ind, aggregated_hist_R, 0.3, color='red', label='Red')
plt.bar(ind, aggregated_hist_G, 0.3, color='green', bottom=aggregated_hist_R, label='Green')
plt.bar(ind, aggregated_hist_B, 0.3, color='blue', bottom=(aggregated_hist_R + aggregated_hist_G), label='Blue')

plt.legend()
plt.savefig('aggregated_RGB_histogram_bar_chart.png')
