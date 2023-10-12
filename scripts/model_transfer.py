import os
import shutil

# Define the source directory and the destination directory
src_dir = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/GOWT_Inverse/cycle_gan'
dst_dir = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/staining_results/GOWT_Inverse'  # Or another destination directory

# Loop through the numbers 5 to 60 in increments of 5
for i in range(5, 65, 5):
    epoch_dir_name = f'Epoch{i}'
    epoch_dir_path = os.path.join(dst_dir, epoch_dir_name)
    
    # Create the sub-directory if it doesn't exist
    if not os.path.exists(epoch_dir_path):
        os.makedirs(epoch_dir_path)

    # File names to look for
    files_to_copy = [f"{i}_net_D_A.pth", f"{i}_net_D_B.pth", f"{i}_net_G_A.pth", f"{i}_net_G_B.pth"]
    
    for file_name in files_to_copy:
        src_file_path = os.path.join(src_dir, file_name)
        
        # Check if the file exists before attempting to copy it
        if os.path.exists(src_file_path):
            new_file_name = file_name.replace(f"{i}_", "latest_")
            dst_file_path = os.path.join(epoch_dir_path, new_file_name)
            
            # Copy the file
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Copied {src_file_path} to {dst_file_path}")
