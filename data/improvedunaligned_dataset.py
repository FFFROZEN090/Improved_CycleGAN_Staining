import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


class ImprovedUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with Ground Truth of Gray Scale images.

    It requires two directories to host training images from domain A '/path/to/data/trainA', domain A GT '/path/to/data/trainA_GT'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA', '/path/to/data/testA_GT' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_A_GT = os.path.join(opt.dataroot, opt.phase + 'A_GT')  # create a path '/path/to/data/trainA_GT'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_GT_paths = sorted(make_dataset(self.dir_A_GT, opt.max_dataset_size))    # load images from '/path/to/data/trainA_GT'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_GT_size = len(self.A_GT_paths)  # get the size of dataset A_GT

        btoA = self.opt.direction == 'BtoA'

        # Exit if direction is BtoA
        if btoA:
            print("Direction BtoA is not supported for ImprovedUnalignedDataset")
            exit()
        
        self.A_tf_type = 'grayscale'
        self.B_tf_type = 'RGB'
        self.A_GT_tf_type = 'RGB'
        self.transform_A = get_transform(self.opt, grayscale=self.A_tf_type)
        self.transform_B = get_transform(self.opt, grayscale=self.B_tf_type)
        self.transform_A_GT = get_transform(self.opt, grayscale=self.A_GT_tf_type)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_GT_path = self.A_GT_paths[index % self.A_GT_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('RGB')
        # Transform the image B to numpy array
        B_np = np.array(B_img)

        # Transpose the image B
        B_np = B_np.transpose(2, 0, 1)

        A_GT_img = Image.open(A_GT_path).convert(self.A_GT_tf_type)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        A_GT = self.transform_A_GT(A_GT_img)

        return {'A': A, 'B': B, 'A_GT':A_GT, 'A_paths': A_path, 'B_paths': B_path, 'A_GT_paths': A_GT_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    """ 
    Visualize the dataset images for debugging
    Parameters:
        A dictionary containing the images and their paths

    Returns:
        None
    """
    def visualize(self, data):
        # Initialize the figure with grid of 1x3
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Display the images
        axs[0].imshow(data['A'])
        axs[0].set_title('Image A')
        axs[0].axis('off')

        axs[1].imshow(data['B'])
        axs[1].set_title('Image B')
        axs[1].axis('off')

        axs[2].imshow(data['A_GT'])
        axs[2].set_title('Image A GT')
        axs[2].axis('off')

        # Initialize the figure saving path
        save_path = 'data_visualization.png'

        # Save the figure
        plt.savefig(save_path)
