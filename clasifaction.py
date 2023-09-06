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


def solider_model():
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.MODEL.TEST_WEIGHT =  './model/transformer_120.pth'
    cfg.merge_from_file("./configs/market/swin_base.yml")

    #Crea el modelo y carga los pesos
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    return model

# En una carpeta voy a tener fotos de 2 personas. Las debe clasificar correctamente
def k_mean_simple():
    input_dir = 'people/simple'
    glob_dir = input_dir + '/*.png'
    paths = [file for file in sorted(glob.glob(glob_dir))]

    # Load and resize images
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in sorted(glob.glob(glob_dir))]
    # images = np.array(np.float32(images).reshape(len(images), -1)/255)
    images = np.array(np.float32(images).transpose(0, 1, 2, 3))

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    model = solider_model()
    model.eval()
    with torch.no_grad():
        s_frame = [transform(img) for img in images]
        features,feature_maps = model(torch.stack(s_frame,dim=0),cam_label=0, view_label=0) #Genera embedding


    # Use MobileNetV2 for feature extraction
    # model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
    # features = model.predict(images.reshape(-1, 224, 224, 3))

    reshaped_features = features.reshape(images.shape[0], -1)
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
    if os.path.exists('people/simple/output'):
        print('Limipeza')
        shutil.rmtree('people/simple/output')
    for i in range(k):
        os.makedirs("people/simple/output/cluster" + str(i))
    for i in range(len(paths)):
        shutil.copy2(paths[i], "people/simple/output/cluster" + str(kpredictions[i]))


# Voy a tener mas de 6 personas clasificar en 6 personas
def k_mean_simple_varios():
    pass

# Voy a tener muchas imagenes y el se va encargar de saber que K es optimo
def k_means_optimize_K():
    pass

k_mean_simple()