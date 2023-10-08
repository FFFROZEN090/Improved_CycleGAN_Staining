import cv2
print(cv2.__version__)

# Load the TIFF image
image = cv2.imread('/home/jli/Cell_cycleGAN/stain_postprocess/original_data/01_1_fake_stained.tif', cv2.IMREAD_COLOR)

# Convert the image to LAB color space for color correction
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Split the LAB image into its channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Apply histogram equalization to the L channel for contrast enhancement
l_channel_eq = cv2.equalizeHist(l_channel)

# Merge the equalized L channel with the original A and B channels
lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))

# Convert the LAB image back to BGR color space
enhanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_Lab2BGR)

# Save the enhanced image to a file
cv2.imwrite('enhanced_image.tif', enhanced_image)