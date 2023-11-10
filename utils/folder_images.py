import os, shutil
import random

def select_and_copy_random_images(root_folder):
    # Walk through all the subdirectories in the root folder
    for subdir, _, files in os.walk(root_folder):
        # Skip the root itself
        if subdir == root_folder:
            continue

        # Get the number of the parent folder
        parent_folder_number = os.path.basename(subdir)

        # Create a new parent folder with the prefix '2-'
        new_parent_folder = os.path.join(root_folder, f"2-{parent_folder_number}")
        os.makedirs(new_parent_folder, exist_ok=True)
        
        # If there are fewer than 6 images, take as many as available
        selected_files = random.sample(files, min(6, len(files)))

        # Copy the selected images to the new parent folder
        for file in selected_files:
            original_file_path = os.path.join(subdir, file)
            new_file_path = os.path.join(new_parent_folder, file)
            
            shutil.copy2(original_file_path, new_file_path)
            print(f"Copied {file} to {new_file_path}")





def rename_and_copy_images(root_folder):
    # Create the destination folder if it doesn't exist
    dest_folder = os.path.join(root_folder, 'query')
    os.makedirs(dest_folder, exist_ok=True)
    
    # Walk through all the subdirectories in the root folder
    for subdir, _, files in os.walk(root_folder):
        # Skip the root and destination folder itself
        if subdir == root_folder or subdir == dest_folder:
            continue
        
        # Get the number of the parent folder
        parent_folder_number = os.path.basename(subdir)
        
        # Process each file in the subdirectory
        for file in files:
            # Construct the new file name
            random_number = str(random.randint(0, 999999)).zfill(6)
            new_file_name = f"{parent_folder_number}_c1s1_{random_number}_01.jpg"
            
            # Full paths for original and new file
            original_file_path = os.path.join(subdir, file)
            new_file_path = os.path.join(dest_folder, new_file_name)
            
            # Copy and rename the image to the destination folder
            shutil.copy2(original_file_path, new_file_path)
            print(f"Copied and renamed {file} to {new_file_path}")