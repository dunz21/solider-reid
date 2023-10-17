from deepface import DeepFace
from utils.colab import extract_images_from_subfolders,plot_mds
import os, shutil





if __name__ == "__main__":
    folders = [
        # './images_subframev2',
        './images_subframev2/29',
        './images_subframev2/48',
        # './images_subframev2/36',
        # './images_subframev2/90',
        # './images_subframev2/130',
        # './images_subframev2/139',
        # './images_subframev2/89',
        # './images_subframev2/6',
        # './images_subframev2/45',
        # './images_subframev2/108',
        # './images_subframev2/158',
        # './images_subframev2/10',
        # './images_subframev2/44',
        # './images_subframev2/72',
        # './images_subframev2/2',
        # './images_subframev2/90',
        # './images_subframev2/77',
        # './images_subframev2/130',
        # './images_subframev2/2',
        # './images_testsss',
        ]
    features, images_names = face_id(folder_path=folders)
    plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='Face')
    