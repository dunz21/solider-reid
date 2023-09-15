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

def optimal_k_with_plot(features, save_path):
    distortions = []
    K = range(1, 40)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(features)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')

    plt.savefig(save_path+'.png')

    best_k = 3
    return best_k

def optimal_k_with_silhouette(features, max_clusters=40, save_path=None):
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)  # Start from 2 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the K value with the highest silhouette score
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]

    if save_path:
        # Plot the silhouette scores and save as an image
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal K')
        plt.grid()
        plt.savefig(save_path + '_silhouette.png')

    return best_k

def optimal_k(features):
    distortions = []
    K = range(1, 60)  # Check for up to 10 clusters
    for k in K:
        kmeanModel = KMeans(n_clusters=k,n_init=10)
        kmeanModel.fit(features)
        distortions.append(kmeanModel.inertia_)

    deltas = np.diff(distortions)
    
    double_deltas = np.diff(deltas)
    
    best_k = np.where(double_deltas > 0)[0][0] + 2  # +2 because the index is shifted due to double differentiation
    best_k=24
    return best_k


def solider_model():
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.TEST.WEIGHT =  './model/swin_base_market.pth'

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    return model

def main(path_folder):
    base_dir = path_folder
    paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir) for f in filenames if f.endswith('.png')]
    paths = sorted(paths)
    features = transform_images_solider(paths)
    k_mean_simple(features, paths, base_dir)
    
def transform_images_solider(path_list):
    images = [np.array(np.float32(cv2.imread(file))) for file in path_list]
    model = solider_model()

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Resize((384,128), antialias=True),
            pth_transforms.Normalize((0.485, 0.456, 0.606), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        s_frame = [transform(img) for img in images]
        features,feature_maps = model(torch.stack(s_frame,dim=0),cam_label=0, view_label=0) #Genera embedding
    return features

def k_mean_simple(features, paths, base_dir, hash=""):
    # reshaped_features = features.reshape(len(paths), -1)
    # Normalize the features
    # normalized_features = normalize(reshaped_features, axis=1)
    # Reduce dimensions using PCA
    # pca = PCA(n_components=0.95)
    # reduced_features = pca.fit_transform(normalized_features)
    # Cluster using KMeans
    # k = optimal_k(reduced_features)
    k = 96
    # optimal_k_with_plot(reduced_features,base_dir)
    # optimal_k_with_silhouette(features=reduced_features,save_path=base_dir)
    kmodel = KMeans(n_clusters=k, random_state=728, n_init=10)
    kmodel.fit(features)
    kpredictions = kmodel.predict(features)

    # Save clustered images to separate folders
    result_dir = base_dir + '_result_kmeans_'+str(k)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    for i in range(k):
        os.makedirs(result_dir + "/cluster" + str(i))
    for i in range(len(paths)):
        destination_folder = result_dir + "/cluster" + str(kpredictions[i])
        shutil.copy2(paths[i], destination_folder)

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


# for i in range(1,2):
#     main('images_copy_background_isnet-general-use')

main('images_copy')

# main('images_copy_background_isnet-general-use')
# main('images_copy_background_silueta')
# main('images_copy_background_u2net')
# main('images_copy_background_u2net_human_seg')
# main('images_copy_background_u2netp')