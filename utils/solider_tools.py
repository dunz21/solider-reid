from utils.colab import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector
from utils.pretools import load_model as load_model_solider,preprocess_image


#Probar normalizar datos. Aplicar clustering antes!!! Que puta es el 1024, y aplicar medidas locales
#### MOVER DE ACA #####
#### MOVER DE ACA #####
#### MOVER DE ACA #####
#### MOVER DE ACA #####
def extract_images_from_subfolders(folder_paths):
    # If the input is a string (single folder path), convert it into a list
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    all_images = []
    
    for folder_path in folder_paths:
        # Walk through each main folder and its subfolders
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # For each subfolder, find all .png images
            images = glob.glob(os.path.join(dirpath, '*.png'))
            all_images.extend(images)
    return all_images

def solider_result(folder_path="", weight='',semantic_weight=0.2):
    model = load_model_solider(weight=weight,semantic_weight=semantic_weight)
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,384,128)], dim=0) for img in images]
    with torch.no_grad():
        features_list, _ = model(torch.cat(total_batch,dim=0))
    
    # Convert tensor to numpy array
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def create_dataframe_from_folder(folder_path, csv_file_name="output.csv", weight=''):
    data = []  # Initialize a list to hold all the data

    # Filter out .DS_Store and sort the remaining folder names numerically
    sorted_folders = sorted(
        filter(lambda f: f != '.DS_Store' and f.isdigit(), os.listdir(folder_path)),
        key=lambda f: int(f)
    )

    # Loop through each sorted and filtered subfolder
    for folder in sorted_folders:
        subfolder_path = os.path.join(folder_path, folder)
        features_array, images_names = solider_result(subfolder_path, weight=weight)
        
        # Process each image in the folder
        for idx, image_name in enumerate(images_names):
            folder_name = image_name.split('_')[1]
            if folder_name:
                # Prepend folder name and image name to the row
                row = [folder_name, image_name] + list(features_array[idx])
                data.append(row)

    # Create column names - ['FolderName', 'ImageName', '1', '2', ..., '1024']
    columns = ['FolderName', 'ImageName'] + [str(i) for i in range(1, 1025)]

    # Create a DataFrame using the data and the column names
    df = pd.DataFrame(data, columns=columns)

    # Get the current working directory
    current_working_dir = os.getcwd()

    # Set the CSV file path to the current working directory
    csv_file_path = os.path.join(current_working_dir, csv_file_name)

    # Export the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    return df