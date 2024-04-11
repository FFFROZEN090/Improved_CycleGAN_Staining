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
    original image use KL divergence for each color in HSV space.
"""

import cv2
import numpy as np
import os
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy.linalg import sqrtm
from PIL import Image
import argparse






'''
    This function is used to evaluate the stained image with the ground truth.
    The principle is to convert the stained image into binary image and compare to ground truth pixel wise.

'''
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


'''
    This function is used to evaluate the stained image with the original image.
    The principle is to convert the stained image and the original image into grayscale image and compare the grayscale variation.
    Compute MSE, PSNR and SSIM.
'''
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
    
    return psnr


'''
    This function is used to evaluate the stained image with the target image.
    The principle is to calculate the color distribution of the stained image and the target image.
    Convert the color distribution into histogram and compare the histogram using KL divergence.
'''
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

    correlation_loss = distance.correlation(aggregate_hist_stained, aggregate_hist_colorized)

    return correlation_loss


# Function to calculate Frechet Inception Distance (FID)
def calculate_fid(model, real_images, fake_images):
    # Ensure model is in eval mode
    model.eval()
    
    # Preprocess images & calculate features
    def preprocess_images(images):
        processed_images = [TF.resize(img, [299, 299]) for img in images]
        processed_images = torch.stack(processed_images)
        processed_images = (processed_images - 0.5) / 0.5  # Normalize to [-1, 1]
        return processed_images
    
    # Calculate activations
    def get_activations(images):
        with torch.no_grad():
            pred = model(preprocess_images(images))
        return pred.detach().cpu().numpy()
    
    # Preprocess images
    real_images = preprocess_images(real_images)
    fake_images = preprocess_images(fake_images)
    
    # Calculate mean and covariance
    real_activations = get_activations(real_images)
    fake_activations = get_activations(fake_images)
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



def evaluate_image(stained_img, ground_truth_img):
    # Principle 1: Compare stained image with ground truth (using, e.g., mean squared error)
    principle1 = GroundTruthLoss(stained_img, ground_truth_img)

    # Principle 2: Compare grayscale variation (using, e.g., histogram)
    principle2 = GrayscaleLoss(stained_img, ground_truth_img)

    return principle1, principle2


def rescale_values(values, min_value, max_value):
    scaled_values = (values - min_value) / (max_value - min_value)
    return scaled_values


"""
Read Images from a directory with specitic surfix
Parameters:
    dir_path (str) -- the path of the directory
    surfix (str) -- the surfix of the images
Return:
    images (numpy array list) -- a list of images
"""

def read_images(dir_path, surfix):
    images = []
    for file in os.listdir(dir_path):
        if file.endswith(surfix):
            img = Image.open(os.path.join(dir_path, file))
            images.append(img)
    # Convert images to tensor
    images = [TF.to_tensor(img) for img in images]
    return images

def main(stained_img_base_path, ground_truth_img_base_path, colorized_img_base_path, n_epochs=60):
    # Iterate over each epoch
    results_file_path = os.path.join(stained_img_base_path, 'evaluation_results.txt')
    aggregate_hist_colorized = calculate_aggregate_hsv_histogram(colorized_img_base_path)
    # Initialize empty lists to hold metric values
    principle1_means = []
    principle2_means = []
    principle3_values = []
    fid_values = []
    overall_scores = []
    epochs = []


    for epoch in range(5, n_epochs, 5):
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

        # Using FID for Evaluation
        # Calculate FID
        fid_model = inception_v3(pretrained=True)
        fid_model.fc = torch.nn.Identity()
        # Read images
        real_images = read_images(colorized_img_base_path, "tiff")
        fake_images = read_images(epoch_path, "png")

        # Check the size of the images
        if len(real_images) == 0 or len(fake_images) == 0:
            print("No images found in the directory")
            exit()

        fid = calculate_fid(fid_model, real_images, fake_images)


        # Assuming results is a list of tuples where each tuple contains the values of principle1 and principle2 for a single image
        principle1_values = np.array([result[0] for result in results])
        principle2_values = np.array([result[1] for result in results])

        # Rescale principle1, principle2, and principle3 values to the range [0, 1] separately
        principle1_scaled = rescale_values(principle1_values, principle1_values.min(), principle1_values.max())
        principle2_scaled = rescale_values(principle2_values, principle2_values.min(), principle2_values.max())
        # assuming principle3 is an array; if not, replace np.min(principle3) and np.max(principle3) with the actual min and max values
        
        # Now use the rescaled values to compute the mean and overall score
        mean_principle1_scaled = np.mean(principle1_scaled)
        mean_principle2_scaled = np.mean(principle2_scaled)
        
        # (Inside your epoch loop, append metric values)
        epochs.append(epoch)

        principle1_means.append(mean_principle1_scaled)
        principle2_means.append(mean_principle2_scaled)
        principle3_values.append(principle3)
        fid_values.append(fid)




    # Calculate the mean of principle 3
    principle3_means = rescale_values(principle3_values, np.min(principle3_values), np.max(principle3_values))

    # Calculate the overall score
    overall_scores = 0.45 * np.array(principle1_means) + 0.45 * np.array(principle2_means) + 0.1 * np.array(principle3_means)

    # Save the results to a file from Epoch5 to Epoch60
    for i in range(len(epochs)):
        with open(results_file_path, 'a') as f:
            f.write(f'Epoch {epochs[i]}: Principle 1: {principle1_means[i]}, Principle 2: {principle2_means[i]}, Principle 3: {principle3_means[i]}, Overall Score: {overall_scores[i]}, FID Score: {fid_values[i]}\n')
            f.close()




    plt.figure(figsize=(10, 6))

    # Plot each metric
    plt.plot(epochs, principle1_means, label='Accuracy mean', marker='o')
    plt.plot(epochs, principle2_means, label='SSIM', marker='x')
    plt.plot(epochs, principle3_means, label='Color Correlation', marker='s')
    plt.plot(epochs, overall_scores, label='Overall Score', marker='d')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics over Epochs')

    # Add a legend
    plt.legend()

    # Save the plot under stained_img_base_path
    plt.savefig(os.path.join(stained_img_base_path, 'evaluation_metrics.png'))


    # Plot FID in a separate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fid_values, label='FID', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('FID Value')
    plt.title('FID over Epochs')
    plt.legend()

    # Save the plot under stained_img_base_path
    plt.savefig(os.path.join(stained_img_base_path, 'FID.png'))

""" 
# Directory paths
stained_img_base_path = '/home/frozen/CV_FinalProject/Cell_cycleGAN/Evaluation_Dataset/training_dataset_tiledGOWT_Fakulty_Inverse'
ground_truth_img_base_path = '/home/frozen/CV_FinalProject/Cell_cycleGAN/Evaluation_Dataset/GOWT_Inverse/'
colorized_img_base_path = '/home/frozen/CV_FinalProject/Cell_cycleGAN/Training_Datasets/training_dataset_tiledGOWT_Fakulty_Inverse/trainB'

"""
# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the model for the stain transfer task.")

    # The following arguments are required
    parser.add_argument("stained_img_base_path", type=str, help="The base path to the directory containing the stained images.")
    parser.add_argument("ground_truth_img_base_path", type=str, help="The base path to the directory containing the ground truth images.")
    parser.add_argument("colorized_img_base_path", type=str, help="The base path to the directory containing the colorized images.")
    parser.add_argument("n_epochs", type=int, help="The number of epochs to evaluate the model for.")


    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the main function with the directory path
    main(args.stained_img_base_path, args.ground_truth_img_base_path, args.colorized_img_base_path, args.n_epochs)