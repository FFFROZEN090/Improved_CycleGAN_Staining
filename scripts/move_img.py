import os
import shutil

def move_images(src_base_dir, dest_base_dir, epochs):
    """
    Move images from source directory to destination directory.

    Args:
    - src_base_dir: Base source directory containing epoch subdirectories.
    - dest_base_dir: Base destination directory where images will be moved.
    - epochs: List of epoch names like ["Epoch 5", "Epoch 10", ...].
    """
    
    # Image filenames to be moved
    image_names = ["01_1.png", "01_10.png", "01_100.png"]
    
    for epoch in epochs:
        for img_name in image_names:
            src_path = os.path.join(src_base_dir, epoch, img_name)
            
            # Ensure destination epoch directory exists
            dest_epoch_dir = os.path.join(dest_base_dir, epoch)
            os.makedirs(dest_epoch_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_epoch_dir, img_name)
            
            # Move the image
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")
            else:
                print(f"File {src_path} not found!")

# Example usage
src_directory = "/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/training_dataset_tiledGOWT_Fakulty_Inverse/"
dest_directory = "/home/frozen/Report_Cell_CycleGAN/Gray_Inverse_Fakulty"
epochs_list = [f"Epoch{i}" for i in range(5, 65, 5)]
move_images(src_directory, dest_directory, epochs_list)







