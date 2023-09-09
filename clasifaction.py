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

def generate_result_id_md5(result):
    # Convert the result to bytes and calculate its MD5 hash
    result_bytes = torch.tensor(result).numpy().tobytes()  # Serialize the result
    result_hash = hashlib.md5(result_bytes).hexdigest()  # Calculate the MD5 hash

    return result_hash

def generate_result_id(result):
    # Convert the result to bytes and calculate its hash using SHA-256
    result_bytes = pickle.dumps(result)  # Serialize the result
    result_hash = hashlib.sha256(result_bytes).hexdigest()  # Calculate the SHA-256 hash

    return result_hash

def encode_result_base64(result):
    # Serialize the result and encode it as Base64
    result_bytes = torch.tensor(result).numpy().tobytes()  # Serialize the result
    result_base64 = base64.b64encode(result_bytes).decode()  # Encode as Base64 and decode to get a string

    return result_base64

def main():
    # torch.manual_seed(42)

    input_dir = 'people'
    glob_dir = input_dir + '/*.png'
    paths = [file for file in sorted(glob.glob(glob_dir))]
    features = transform_images_solider(paths)
    k_mean_simple(features,paths,input_dir)
    

def transform_images_solider(path_list):
    # Load and resize images
    images = [cv2.resize(cv2.imread(file), (384, 128)) for file in path_list]
    images = np.array(np.float32(images).transpose(0, 1, 2, 3))
    model = solider_model()

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        s_frame = [transform(img) for img in images]
        features,feature_maps = model(torch.stack(s_frame,dim=0),cam_label=0, view_label=0) #Genera embedding
    return features

def transform_image(path):
    image = cv2.resize(cv2.imread(path), (224, 224))
    image = np.array(np.float32(image).reshape(-1,3))
    image = normalize(np.array(image), axis=0)
    return image

def apply_solider(image):
    model = solider_model()
    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        s_frame = transform(image)
        features,feature_maps = model(torch.stack([s_frame],dim=0),cam_label=0, view_label=0) #Genera embedding
    return features

# En una carpeta voy a tener fotos de 2 personas. Las debe clasificar correctamente
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

    # TEST WITH OTHER IMAGE
    # other_image = transform_image('images/5/img_5_20.png')
    # features_other = apply_solider(other_image.reshape(224,224,3))
    # # reduced_features_other = pca.fit_transform(features_other)
    # kpredictions_other = kmodel.predict(features_other)
    # TEST WITH OTHER IMAGE


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



# print_count_unique_ids_in_clusters('./people/output')

for i in range(1,30):
    main()