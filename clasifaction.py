import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from config import cfg
from model import make_model
import torch
from torchvision import transforms as pth_transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datetime import datetime
from utils.metrics import euclidean_distance
import pickle
import hashlib
import base64

def solider_model():
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    # cfg.TEST.WEIGHT =  './model/swin_base.pth'
    cfg.TEST.WEIGHT =  './model/swin_base_market.pth'
    # cfg.TEST.WEIGHT =  './model/transformer_120.pth'
    # cfg.TEST.PRETRAIN_PATH =  './model/swin_base_patch4_window7_224.pth'

    #Crea el modelo y carga los pesos
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    return model

def main():
    input_dir = 'people'
    glob_dir = input_dir + '/*.png'
    paths = [file for file in sorted(glob.glob(glob_dir))]
    features = transform_images_solider(paths)
    k_mean_simple(features,paths,input_dir)
    
def transform_images_solider(path_list):
    images = [np.array(np.float32(cv2.imread(file))) for file in path_list]
    model = solider_model()

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Resize((384,128), antialias=True),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        s_frame = [transform(img) for img in images]
        features,feature_maps = model(torch.stack(s_frame,dim=0),cam_label=0, view_label=0) #Genera embedding
    return features

def k_mean_simple(features,paths,input_dir,hash=""):
    reshaped_features = features.reshape(len(paths), -1)
    # Normalize the features
    normalized_features = normalize(reshaped_features, axis=1)
    # Reduce dimensions using PCA
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(normalized_features)
    # Cluster using KMeans
    k = 2
    kmodel = KMeans(n_clusters=k, random_state=728, n_init=10)
    kmodel.fit(reduced_features)
    kpredictions = kmodel.predict(reduced_features)

    # Save clustered images to separate folders
    if os.path.exists(input_dir+'/output'):
        shutil.rmtree(input_dir+'/output')
    for i in range(k):
        os.makedirs(input_dir+"/output/cluster" + str(i))
    for i in range(len(paths)):
        shutil.copy2(paths[i], input_dir+"/output/cluster" + str(kpredictions[i]))
    print_count_unique_ids_in_clusters(input_dir+'/output',hash)

def print_count_unique_ids_in_clusters(output_folder,hash=""):
    cluster_counts = {}  # Dictionary to store counts for each cluster

    # List all subdirectories (clusters) within the output folder
    subdirectories = [d for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))]

    for cluster in subdirectories:
        cluster_path = os.path.join(output_folder, cluster)
        id_counts = {}  # Dictionary to store counts for each unique ID in the cluster

        for root, _, files in os.walk(cluster_path):
            for filename in files:
                if filename.endswith('.png'):  # Check if the file is a PNG image
                    _, id, _ = filename.split('_')  # Split the filename by underscores
                    if id not in id_counts:
                        id_counts[id] = 0
                    id_counts[id] += 1

        cluster_counts[cluster] = id_counts

    # Get the current date and time as a string
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open('clustering_result.txt', 'a' if os.path.isfile('clustering_result.txt') else 'w') as result_file:
        # Write the date and time followed by the cluster_counts dictionary to the file
        print(f"{current_datetime} {hash} {cluster_counts}", file=result_file)

    return cluster_counts



def apply_solider(path):
    image = cv2.resize(cv2.imread(path), (384, 128))
    model = solider_model()
    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Resize((384,128), antialias=True),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        s_frame = transform(image)
        features,feature_maps = model(torch.stack([s_frame],dim=0),cam_label=0, view_label=0) #Genera embedding
    return features

def compare_2_images():
    # TEST WITH OTHER IMAGE
    features_other = apply_solider('images/3/img_3_20.png')
    features_other2 = apply_solider('images/3/img_3_60.png')
    distance = euclidean_distance(features_other,features_other2)
    print(distance)
    # join_images = np.stack([features_other,features_other2])
    # k = 2
    # kmodel = KMeans(n_clusters=k, random_state=728, n_init=10)
    # kmodel.fit(join_images)
    # kpredictions = kmodel.predict(join_images)
    # TEST WITH OTHER IMAGE


# compare_2_images()

# print_count_unique_ids_in_clusters('./people/output')

for i in range(1,30):
    main()