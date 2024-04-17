

# CycleGAN for Cellular Image Colorization: Technical Documentation

This document provides detailed instructions for setting up and executing the CycleGAN-based cellular image colorization project within a Docker environment. It outlines the creation of a Docker image configured with the necessary dependencies and environment for running the CycleGAN model, as well as steps to deploy a Jupyter Notebook server for interactive development and visualization.



## Docker Environment Setup

### Dockerfile Configuration

The Dockerfile is designed to establish a foundational environment leveraging NVIDIA's PyTorch Docker image. This setup ensures compatibility with CUDA-enabled GPUs for accelerated model training.

### Dockerfile

```bash
# Base image with NVIDIA PyTorch support
FROM nvcr.io/nvidia/pytorch:21.02-py3

# Install additional Python packages
RUN pip install visdom dominate

#Install project-specific Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

#JupyterLab installation and configuration
RUN pip install jupyterlab
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose the default Jupyter Notebook port
EXPOSE 8888

```



### 

### Requirements File

The requirements.txt file specifies the necessary Python libraries and their versions to ensure compatibility and functionality of the CycleGAN model and the data processing scripts.



```bash
--find-links https://download.pytorch.org/whl/torch_stable.html
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
networkx>=2.2
numpy~=1.18.2
pillow~=7.1.1
matplotlib~=3.2.1
opencv-python~=4.2.0.34
requests~=2.23.0
scipy~=1.4.1
scikit-image~=0.19.0
jupyterlab
```

### Docker Image Building

The Docker image is built using the provided Dockerfile, tagging it with the current date to facilitate version control and ensure reproducibility.

To enable rootless operation of Docker, the following configuration script sets up the environment for the current user, allowing Docker commands to be run without root privileges.

```bash
docker build -t cv_final:$(date +%Y-%m-%d) .
Docker Configuration for Rootless Operation

user=$(whoami)

curl -sSL https://get.docker.com/rootless | sh
export PATH=/home/$user/bin:$PATH
export PATH=$PATH:/sbin
export DOCKER_HOST=unix:///run/user/1027/docker.sock
systemctl --user start docker
```





### Running Jupyter Notebook Server in Docker

The final step involves deploying a Docker container configured to run a Jupyter Notebook server. This server provides an interactive environment for developing and testing the CycleGAN model. The command below mounts the project directory within the container, exposes the Jupyter Notebook server port, and sets various parameters to optimize performance.


```bash
user=$(whoami)
image_name=cv_final
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user
workdir=/home/$user/CV_FinalProject/Cell_cycleGAN

docker run --gpus all --rm -it -w $workdir -v $dir:$dir -p 8888:8888 --shm-size=10g --ulimit memlock=1 --ulimit stack=67108864 $image_id \
jupyter notebook --notebook-dir=$workdir --ip='0.0.0.0' --port=8888 --no-browser --allow-root
```

This command initializes a Jupyter Notebook server within the Docker container, enabling users to access and interact with the project notebooks through a web browser. The configuration facilitates the use of GPU resources for model training, ensuring efficient execution of deep learning tasks.



## Dataset Preprocessing

This section outlines the procedures for downloading and preprocessing the datasets used in our research.



### Dataset Downloading



#### Data Acquisition

Our research utilizes two primary datasets, selected for their relevance and impact on the field of cell study:

1. **HeLa Dataset**: Sourced from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/), this dataset is crucial for its focus on the HeLa cell line, a key element in cellular and cancer research. The dataset's extensive application has greatly advanced our understanding of cell behavior.
2. **GOWT Dataset**: Notable for its clearly defined cell shapes, which are ideal for precise segmentation tasks. The distinct morphological features of the GOWT dataset make it an invaluable resource for developing and testing colorization techniques.

Both datasets include segmentation ground truth, which is essential for evaluating the accuracy of our colorization methods.



#### Target Dataset

For real-world benchmarking, we use a dataset from The Cancer Genome Atlas (TCGA), available [here](https://portal.gdc.cancer.gov/analysis_page). We specifically utilize the image `TCGA-CF-A5UA-01Z-00-DX1.7352D4EB-46F5-4EAA-95B9-1D869E8291C3.SVS` for its high-quality, real-world colorized cell imagery, allowing us to compare our results against actual colorized images and ensure realism and applicability in practical analysis.



### Dataset Preprocessing

To enhance dataset size and increase training speed, all images have been cropped to 128x128 pixels.

For the GOWT dataset, observations indicated that for consistency with the target dataset, the cell regions should appear darker, while the background should be lighter. Thus, pixel values have been inverted under the uint8 format.

Furthermore, to support the evaluation of new training losses proposed in this project, all labeled images have been converted to a binary format, consisting only of the values 255 (white) and 0 (black).

All relevant scripts can be found in the `scripts` directory.



### Dataset Preparation

In accordance with the requirements of the CycleGAN model, all datasets are organized in the following structure within one directory:

- `trainA`: Domain A training set images
- `trainB`: Domain B training set images
- `testA`: Domain A testing set images
- `testB`: Domain B testing set images

Additionally, new losses introduced in the CycleGAN training necessitate pairing segmentation ground truth with both `trainA` and `testA` images. This data is stored in the directories `trainA_GT` and `testA_GT`, each containing binary labels.





## Project Running



### Training Run

The `train.py` script contains the training code. All essential parameters are specified below. This command initializes the training process using specified directories and settings:

```bash
python train.py --dataroot {path_to_trainA_and_trainB} --results_dir {path_to_results} --name {name_of_experiment} --load_size {load_size} --crop_size {crop_size} --pool_size {pool_size} --model {model_name}
```



### Testing Run

After training, it's necessary to evaluate the model's performance across different epochs due to the typical challenges in unpaired colorization, where the loss function may only converge to a general score that does not reflect localized accuracy. To address this, we propose an evaluation method that involves testing the model across all epochs:

```bash
python test.py --dataroot {dataroot} --name {epoch_dir_path} --no_dropout --model {model_name} --direction {direction} --input_nc {input_nc} --output_nc {output_nc} --dataset_mode {dataset_mode}
```

Results and visual outputs can be reviewed in the specified results directory.



### Evaluation Run

Post-testing, the selected colorized images should be organized into structured directories for further evaluation. Our evaluation process is capable of scoring multiple epochs to determine the most effective training outcomes.

Execute the evaluation script using the following command:

```bash
python evaluation/Stain_Evaluation.py
```

This script provides evaluation scores and insights into the overall performance of the model.



## New Model and Losses



### New Dataloader

A custom dataloader is implemented in `data/improvedunaligned_dataset.py` to pair input images with corresponding Ground Truth (GT) images during training and testing phases. This dataloader handles unaligned datasets by loading images from separate directories for each domain and their GT counterparts. Key features of this dataloader include: - Loading grayscale GT images. - Handling both training and testing data directories. - Enforcing specific transformations based on the image type (grayscale for input and RGB for GT). - Optionally reversing the direction of image pairing (not supported in current configuration).

```python
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


        params = {'flip': random.random() > 0.5, 'crop_pos': (0, 0)}
        
        self.A_tf_type = 'grayscale'
        self.B_tf_type = 'RGB'
        self.A_GT_tf_type = 'RGB'
        self.transform_A = get_transform(self.opt, params=params, grayscale=self.A_tf_type)
        self.transform_B = get_transform(self.opt, grayscale=self.B_tf_type)
        self.transform_A_GT = get_transform(self.opt, params=params, grayscale=self.A_GT_tf_type)

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
```



### New Model

Enhancements to the model are specified in `models/seg_cycle_gan_model.py`. This includes integration of new loss functions tailored to improve colorization accuracy and visual consistency. The model combines several loss components: - Traditional GAN losses for both domains. - Cycle consistency losses to ensure the input and reconstructed images are similar. - Identity losses to preserve color of input images when translated back and forth. - Custom losses for improved fidelity to the ground truth, better color variation, and hue, saturation, and value (HSV) accuracy.

```python
self.loss_GT_fake = improved_losses.GT_Loss(self.fake_A, self.real_A_GT)
self.loss_color_variation_fake = improved_losses.ColorVariation_Loss(self.real_A, self.fake_B)
self.loss_hsv_fake = improved_losses.HSV_Loss(self.fake_B, self.real_B)

self.loss_GT_rec = improved_losses.GT_Loss(self.rec_A, self.real_A_GT)
self.loss_color_variation_rec = improved_losses.ColorVariation_Loss(self.real_A, self.rec_B)
self.loss_hsv_rec = improved_losses.HSV_Loss(self.rec_B, self.real_B)


# combined loss and calculate gradients
self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + 5 * self.loss_GT_rec + 5 * self.loss_color_variation_rec + 10 * self.loss_hsv_rec + 5 * self.loss_GT_fake + 5 * self.loss_color_variation_fake + 10 * self.loss_hsv_fake
```



### New Losses

New loss functions are defined in `models/improved_losses.py` to address specific challenges in unpaired image colorization: - **GT Loss**: Measures the pixel-wise accuracy between the colorized image and the grayscale GT. - **Color Variation Loss**: Encourages preservation of color dynamics by comparing the structural similarity index (SSIM) between the original and colorized images. - **HSV Loss**: Assesses the similarity in color distribution in the HSV space, crucial for maintaining perceptual color consistency. These losses are particularly designed to optimize the colorization process by focusing on detailed aspects of image quality and fidelity, significantly enhancing the model's performance in realistic scenarios. 

```python
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

```



## All Detailed Training Results and Usage Example Can Be Seen in `demo.ipynb`
