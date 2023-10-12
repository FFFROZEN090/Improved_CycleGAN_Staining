"""
    This is the evaluation module for the stain transfer task. After training the
model, we can use this module to evaluate the model for the stain transfer task.
There are theory principles for evaluating the model: 

1.  Converte stained image
    into binary image. Compare the stained image with the ground truth image. To
    test the outlierness of the staining result. 

2.  Compare the staining result with
    the original image. To test the similarity of the grayscale variation. 

3.  Statistical comparison of the color distribution of the staining result and the
    original image use KL divergence for each color in RGB space.
"""

import cv2
import numpy as np
import os
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim




# Directory paths
stained_img_base_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/'
ground_truth_img_base_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/'
colorized_img_base_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/training_dataset_GOWT_Inverse/trainB/'
binary_convert_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/Binary_Convert/'

def GroundTruthLoss(stained_img, ground_truth_img):

    # Convert RGB image to binary image
    stained_img_gray = cv2.cvtColor(stained_img, cv2.COLOR_BGR2GRAY)
    ground_truth_img_gray = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2GRAY)
    # Use Otsu's method to convert the image to binary image
    _, stained_img_gray = cv2.threshold(stained_img_gray, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, ground_truth_img_gray = cv2.threshold(ground_truth_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Invert the binary image
    stained_img_gray = cv2.bitwise_not(stained_img_gray)

    # Pixel wise comparison between stained image and ground truth image
    union_score = np.sum(np.logical_and(stained_img_gray, ground_truth_img_gray))
    total_score = np.sum(np.logical_or(stained_img_gray, ground_truth_img_gray))

    # Calculate the loss
    loss = union_score / total_score

    # Return the loss
    return loss

def GrayscaleLoss(stained_img, original_img):
    # Convert to grayscale
    stained_gray = cv2.cvtColor(stained_img, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Compute MSE
    mse = np.sum((stained_gray - original_gray) ** 2) / float(stained_gray.shape[0] * stained_gray.shape[1])
    
    # Compute PSNR
    if mse == 0:
        psnr = 100  # identical images
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Compute SSIM
    ssim_value = ssim(stained_gray, original_gray)
    
    return ssim_value

def calculate_aggregate_hsv_histogram(img_dir):
    aggregate_hist = np.zeros((256 * 3,))

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for channel in range(3):
            hist_channel = cv2.calcHist([img_hsv], [channel], None, [256], [0, 256]).flatten()
            aggregate_hist[channel * 256:(channel + 1) * 256] += hist_channel

    return aggregate_hist / aggregate_hist.sum()  # Normalize

def ColorLoss(stained_img_dir, aggregate_hist_colorized):
    aggregate_hist_stained = calculate_aggregate_hsv_histogram(stained_img_dir)

    # Normalize the histograms
    aggregate_hist_stained /= aggregate_hist_stained.sum()
    aggregate_hist_colorized /= aggregate_hist_colorized.sum()

    chi_square_loss = 0.5 * np.sum(((aggregate_hist_stained - aggregate_hist_colorized) ** 2) / (aggregate_hist_stained + aggregate_hist_colorized + 1e-6))
    emd_loss = wasserstein_distance(aggregate_hist_stained, aggregate_hist_colorized)
    correlation_loss = distance.correlation(aggregate_hist_stained, aggregate_hist_colorized)

    return correlation_loss



def evaluate_image(stained_img, ground_truth_img):
    # Principle 1: Compare stained image with ground truth (using, e.g., mean squared error)
    principle1 = GroundTruthLoss(stained_img, ground_truth_img)

    # Principle 2: Compare grayscale variation (using, e.g., histogram)
    principle2 = GrayscaleLoss(stained_img, ground_truth_img)

    return principle1, principle2


# Iterate over each epoch
results_file_path = os.path.join(stained_img_base_path, 'evaluation_results.txt')
aggregate_hist_colorized = calculate_aggregate_hsv_histogram(colorized_img_base_path)
# Initialize empty lists to hold metric values
principle1_means = []
principle2_means = []
principle3_means = []
overall_scores = []
epochs = []
try:    
    for epoch in range(5, 65, 5):
        epoch_path = os.path.join(stained_img_base_path, f'Epoch{epoch}')
        print(f'Starting evaluation for epoch {epoch}')

        if not os.path.exists(epoch_path):
            print(f"Epoch directory not found: {epoch_path}")
            continue

        results = []
        num_images = 0

        for img_name in os.listdir(epoch_path):

            stained_img_path = os.path.join(epoch_path, img_name)
            GT_img_name = img_name.replace('.png', '.tiff')
            ground_truth_img_path = os.path.join(ground_truth_img_base_path, 'GT', GT_img_name)

            stained_img = cv2.imread(stained_img_path)
            ground_truth_img = cv2.imread(ground_truth_img_path)

            if stained_img is None or ground_truth_img is None:
                print(f'Error reading image: {stained_img_path} or {ground_truth_img_path}')
                continue

            principle1, principle2 = evaluate_image(stained_img, ground_truth_img)
            results.append((principle1, principle2))
            num_images += 1

        mean_results = np.mean(results, axis=0)
        principle3 = ColorLoss(epoch_path, aggregate_hist_colorized)

        # Create results figure
        # (Inside your epoch loop, append metric values)
        principle1_means.append(mean_results[0])
        principle2_means.append(mean_results[1])
        principle3_means.append(principle3)
        overall_scores.append(0.4 * mean_results[0] + 0.55 * mean_results[1] + 0.05 * principle3)
        epochs.append(epoch)


        # Open results file in append mode ('a')
        with open(results_file_path, 'a') as f:
            f.write(f'\nEpoch: {epoch}\n')
            f.write(f'Principle 1 mean: {mean_results[0]}\n')
            f.write(f'Principle 2 mean: {mean_results[1]}\n')
            f.write(f'Principle 3 mean: {principle3}\n')
            f.write(f'Overall Score: {0.4 * mean_results[0] + 0.55 * mean_results[1] + 0.05 * principle3}\n')

            print(f'Finished evaluation for epoch {epoch}')

except Exception as e:
    print(f"An error occurred: {e}")


plt.figure(figsize=(10, 6))

# Plot each metric
plt.plot(epochs, principle1_means, label='Principle 1 mean', marker='o')
plt.plot(epochs, principle2_means, label='Principle 2 mean', marker='x')
plt.plot(epochs, principle3_means, label='Principle 3 mean', marker='s')
plt.plot(epochs, overall_scores, label='Overall Score', marker='d')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Evaluation Metrics over Epochs')

# Add a legend
plt.legend()

# Save the plot
plt.savefig('/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/evaluation_metrics.png', format='png')