import os
import filecmp
import shutil

def compare_folders_and_move_diffs(folder1, folder2, target_folder):
    # Check if the given paths are directories
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        return "Both paths should be directories."

    # Initialize a dircmp object to compare the folders
    dcmp = filecmp.dircmp(folder1, folder2)

    # Lists to store the differences found in files
    left_only = dcmp.left_only
    right_only = dcmp.right_only
    diff_files = dcmp.diff_files

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Move differing files to the target folder
    for filename in left_only:
        source_path = os.path.join(folder1, filename)
        target_path = os.path.join(target_folder, filename)
        shutil.move(source_path, target_path)

    # Format and return the results
    result = {
        "Files only in folder1": left_only,
        "Files only in folder2": right_only,
        "Differing files": diff_files
    }

    return result

# Example usage:
folder1 = "/home/diego/Downloads/DataTrainingRevisada"
folder2 = "/home/diego/Downloads/DataTrainingRevisada/DataTrainingRevisada"
target_folder = "/home/diego/Downloads/DataAdicionalTest"

differences = compare_folders_and_move_diffs(folder1, folder2, target_folder)
for key, value in differences.items():
    print(key)
    for item in value:
        print(f"  {item}")

print(f"Differing files have been moved to {target_folder}.")
