""" 
Improved Losses for Image Colorization Specified for segmentation tasks

New Losses:
    - Loss for Colorized images compare to the Ground Truth
    - Loss for Colorized images colorvariation after convert to Gray Scale
    - Distance for Colorized images and Target images in HSV space

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


def denormalize(tensor):
    """
    Denormalize the tensor image
    :param tensor: Tensor image
    :return: Denormalized tensor image
    """
    return ( (tensor + 1.0) / 2.0) * 255


def RGBBin2Bin(tensor):
    """
    Convert RGB binary tensor batch to binary tensor batch
    :param tensor: RGB binary tensor with shape (B, 3, H, W)
    :return: Binary tensor with shape (B, 1, H, W)
    """
    # Take the first channel of the tensor and keep the shape as (B, 1, H, W)
    return tensor[:, 0:1, :, :]


def gray2binary(images_A, threshold=127):
    """
    Convert grayscale images tensor to binary images tensor
    :param image: Grayscale image tensor
    :param threshold: Threshold value
    :return: Binary image
    """
    
    # Check the shape of the images whether contains one color channel
    if images_A.shape[1] != 1:
        raise ValueError("The input image must be grayscale image")
    
    # Convert the grayscale images to binary values 0 or 255
    images_A = (images_A > threshold).float() * 255
    return images_A


def GT_Loss(images_A, images_GT):
    """
    Calculate the Ground Truth Loss
    :param images_A: Colorized images torch tensor
    :param images_B: Ground Truth images torch tensor
    :return: Ground Truth Loss
    """
    

    # Create temporary directory to save the images
    if not os.path.exists("temp"):
        os.makedirs("temp")


    # Convert images_A to binary images use the threshold 127
    images_A = denormalize(images_A)
    images_A = gray2binary(images_A)

    # Convert images B from RGB formated Binary images to Binary images
    images_GT = RGBBin2Bin(images_GT)
    images_GT = denormalize(images_GT)

    # Check if images_A and Images_B have the same shape
    if images_A.shape != images_GT.shape:
        raise ValueError("The input images must have the same shape")
    
    # Check if the images only contain pixel values between 0 and 255
    if images_A.min() < 0 or images_A.max() > 255:
        raise ValueError("The input images A must have pixel values between 0 or 255")
    if images_GT.min() < 0 or images_GT.max() > 255:
        raise ValueError("The input images GT must have pixel values between 0 or 255")
    
    # Calculate the Ground Truth Loss for this batch
    loss = torch.sum(torch.abs(images_A - images_GT))

    # Nornalize the loss value by calculate the portion of the loss value to the theortical maximum loss value
    max_loss = 255 * (images_A.shape[0] * images_A.shape[2] * images_A.shape[3])
    loss = loss / max_loss


    return loss



def ColorVariation_Loss(images_Real, images_Fake):
    """
    Calculate the Color Variation Loss
    :param images_Real: Original images torch tensor
    :param images_Fake: Colorized images torch tensor
    :return: Color Variation Loss
    """
    # Check if images_Real and Images_Fake have the same batch size
    if images_Real.shape[0] != images_Fake.shape[0]:
        raise ValueError("The input images must have the same batch size")

    # Permute the images to (B, H, W, C)
    images_Real = images_Real.permute(0, 2, 3, 1)
    images_Fake = images_Fake.permute(0, 2, 3, 1)

    # For Real images, squeeze the channel dimension
    if images_Real.shape[3] == 1:
        images_Real = images_Real.squeeze(3)

    # Loop through the batch of images
    loss = 0
    for i in range(images_Real.shape[0]):
        
        # Convert Fake images to Gray Scale
        fake_gray = cv2.cvtColor(denormalize(images_Fake[i].cpu().detach().numpy()).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Convert Real images to Gray Scale
        real_gray = denormalize(images_Real[i].cpu().detach().numpy()).astype(np.uint8)

        # Compute ssim value
        loss += ( ssim(fake_gray, real_gray, data_range=255) + 1 ) / 2

    # Calculate the average loss value
    loss = 1 - loss / images_Real.shape[0]

    return loss


def HSV_Loss(images_A, images_B):
    """
    Calculate the HSV Loss
    :param images_A: Colorized images torch tensor
    :param images_B: Ground Truth images torch tensor
    :return: HSV Loss
    """
    # Check if images_A and Images_B have the same batch size
    if images_A.shape[0] != images_B.shape[0]:
        raise ValueError("The input images must have the same batch size")
    
    # Permute the images to (B, H, W, C)
    images_A = images_A.permute(0, 2, 3, 1)
    images_B = images_B.permute(0, 2, 3, 1)

    # Loop through the batch of images
    loss = 0
    for i in range(images_A.shape[0]):
        # Convert images A to numpy array
        img_A = denormalize(images_A[i].cpu().detach().numpy()).astype(np.uint8)

        # Convert images B to numpy array
        img_B = denormalize(images_B[i].cpu().detach().numpy()).astype(np.uint8)

        # Convert images A to HSV space
        img_A = cv2.cvtColor(img_A, cv2.COLOR_RGB2HSV)

        # Convert images B to HSV space
        img_B = cv2.cvtColor(img_B, cv2.COLOR_RGB2HSV)

        # Calculate the histogram of images A
        hist_A = cv2.calcHist([img_A], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

        # Calculate the histogram of images B
        hist_B = cv2.calcHist([img_B], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Normalize the histogram values
    hist_A = hist_A / hist_A.sum()
    hist_B = hist_B / hist_B.sum()

    # Calculate Correlation Coefficient
    loss = 1 - ( distance.correlation(hist_A.flatten(), hist_B.flatten()) + 1 ) / 2

    return loss


