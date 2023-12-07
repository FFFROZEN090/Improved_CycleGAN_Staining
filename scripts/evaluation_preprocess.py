import os
import shutil

# Source directory path
source_base_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/staining_results/training_dataset_tiledGOWT_Fakulty_Inverse_valA/'

# Target directory path
target_base_path = '/home/frozen/Experiments_Repitition/Cell_cycleGAN/Evaluation_Dataset/training_dataset_tiledGOWT_Fakulty_Inverse_valA'

# Iterate over each epoch folder
for epoch in range(5, 65, 5):  # Starts at Epoch5 and ends at Epoch60, incrementing by 5
    source_epoch_path = os.path.join(source_base_path, f'Epoch{epoch}', 'test_latest', 'images')
    target_epoch_path = os.path.join(target_base_path, f'Epoch{epoch}')
    
    # Create the corresponding target epoch directory if it doesn't exist
    if not os.path.exists(target_epoch_path):
        os.makedirs(target_epoch_path)
    
    # Iterate over each file in the source epoch directory
    for filename in os.listdir(source_epoch_path):
        if filename.endswith('_fake.png'):
            # Extract the original part of the name
            original_part = filename.split('_')[1]
            index_part = filename.split('_')[2].split('_fake.png')[0]

            # Construct the new filename
            new_filename = f'{original_part}_{index_part}.png'

            # Construct the full path for source and target
            source_file_path = os.path.join(source_epoch_path, filename)
            target_file_path = os.path.join(target_epoch_path, new_filename)

            # Copy the file
            shutil.copy2(source_file_path, target_file_path)