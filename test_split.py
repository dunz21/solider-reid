#from deepface import DeepFace
#from utils.colab import extract_images_from_subfolders,plot_mds
from utils.folder_images import process_dataset,check_and_delete_corrupted_images,find_matches


def check_folder_assert_cam_error():
    #### CHECK FOLDERS ASSERT ERROR ####
    # Paths to the folders
    query_folder = '/home/diego/Documents/solider-reid/datasets/TrainingDatasetSolider/market1501/query'
    gallery_folder = '/home/diego/Documents/solider-reid/datasets/TrainingDatasetSolider/market1501/bounding_box_test'
    # query_folder = '/home/diego/Documents/market1501/query'
    # gallery_folder = '/home/diego/Documents/market1501/bounding_box_test'

    # Perform the check
    matches = find_matches(query_folder, gallery_folder)

    # Output the results
    for query_image, has_match in matches.items():
        if not has_match:
            print(f"No match found in gallery for query image: {query_image}")

    # If you want to assert that every query image must have a match, you can do so
    assert all(matches.values()), "Some query images do not have a match in gallery."
    #### CHECK FOLDERS ASSERT ERROR ####

if __name__ == "__main__":
    process_dataset('/home/diego/Downloads/DataTrainingRevisada')
    #check_and_delete_corrupted_images('/home/diego/Documents/solider-reid/datasets/TrainingDatasetSolider/market1501')






