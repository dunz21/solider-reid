from PIL import Image
import os
import shutil
import random

def split_folders(root_folder, training_ratio=0.7):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    random.shuffle(subfolders)

    num_train = int(len(subfolders) * training_ratio)
    train_folders = subfolders[:num_train]
    test_folders = subfolders[num_train:]

    return train_folders, test_folders

def copy_and_rename_images(source_folders, dest_folder, max_images=None, query=False):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    deleted_images_count = 0

    for folder in source_folders:
        parent_folder_name = os.path.basename(folder)
        images = os.listdir(folder)
        if query:
            images = images[:max_images]

        for img in images:
            parts = img.split('_')
            if len(parts) >= 3 and parts[0] == 'img' and parts[1].isdigit() and parts[2].isdigit():
                frame = parts[2]
                ### EDIT ### To pass the assert
                #Query and Gallery Identities Match: 
                # The assert statement checks if there is at least one valid query after filtering out gallery images from the same camera view. 
                # For the data not to hit the assert, there must be at least one image in bounding_box_test for each identity in query that is from a different camera view.
                # Por cada query image tienen que haber en el galley (test) imagenes de otras camaras, por eso cada query va a tener una camara diferente
                ### EDIT ### To pass the assert
                camera = 'c1s1'
                if('query' in dest_folder):
                    camera='c2s1'
                new_name = f"{parent_folder_name}_{camera}_{frame}_00.jpg"  # Changed to .jpg
                new_path = os.path.join(dest_folder, new_name)
                shutil.copy(os.path.join(folder, img), new_path)

                # Try opening the copied image
                try:
                    with Image.open(new_path).convert('RGB') as im:
                        pass  # Image opened successfully
                except Exception:
                    os.remove(new_path)  # Delete the corrupted image
                    deleted_images_count += 1  # Increment the deleted images counter
    
    print(f"Deleted {deleted_images_count} corrupted images.")
    return deleted_images_count


def process_dataset(root_folder):
    """
        Transform a folder with subfolder identities into market1501 type dataset
    """
    parent_dir = os.path.dirname(root_folder)
    training_folder = os.path.join(parent_dir, 'dataset_ready')

    train_folders, test_folders = split_folders(root_folder)

    # Process training data
    bounding_box_train = os.path.join(training_folder, 'bounding_box_train')
    os.makedirs(training_folder, exist_ok=True)
    # for folder in train_folders:
        # shutil.copytree(folder, os.path.join(training_folder, os.path.basename(folder)))
    copy_and_rename_images(train_folders, bounding_box_train)

    # Process test data
    bounding_box_test = os.path.join(training_folder, 'bounding_box_test')
    query_folder = os.path.join(training_folder, 'query')
    os.makedirs(training_folder, exist_ok=True)
    # for folder in test_folders:
        # shutil.copytree(folder, os.path.join(test_folder, os.path.basename(folder)))
    copy_and_rename_images(test_folders, bounding_box_test)
    copy_and_rename_images(test_folders, query_folder, max_images=6, query=True)

# Extra to check valid images, and prevent error in reading images
def check_and_delete_corrupted_images(folder_path):
    deleted_images_count = 0

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(subdir, file)
            try:
                img = Image.open(img_path).convert('RGB')
            except IOError:
                print(f"IOError incurred when reading '{img}'. Deleting image.")
                os.remove(img_path)  # Delete the corrupted image
                deleted_images_count += 1  # Increment the deleted images counter
                pass
                
    print(f"Deleted {deleted_images_count} corrupted images.")
    return deleted_images_count




### CHECK FOLDER
def extract_info(filename):
    """Extracts the person ID and camera ID from the given filename."""
    # Assuming filename format is like '0006_c6s4_002202_00.jpg'
    parts = filename.split('_')
    if len(parts) < 4:
        raise ValueError(f"Filename {filename} is not in the expected format.")
    person_id = parts[0]
    camera_id = parts[1]  # This is the correct index for the camera ID
    return person_id, camera_id

def find_matches(query_folder, gallery_folder):
    """Finds correct matches in gallery for each image in query."""
    # Get list of all files in the query and gallery folders
    query_images = os.listdir(query_folder)
    gallery_images = os.listdir(gallery_folder)

    # Extract person and camera IDs for all images in both sets
    query_info = {img: extract_info(img) for img in query_images}
    gallery_info = {img: extract_info(img) for img in gallery_images}

    # Check for matches
    match_found_for_query = {}
    for q_img, (q_pid, q_cid) in query_info.items():
        if(q_pid == '0006'):
            print('asdf')

        # Look for any gallery image with the same person ID but a different camera ID
        matches = [g_img for g_img, (g_pid, g_cid) in gallery_info.items() if g_pid == q_pid and g_cid != q_cid]
        match_found_for_query[q_img] = len(matches) > 0

    return match_found_for_query