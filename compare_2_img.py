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
from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

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

def apply_solider_v2(path):
    image = cv2.imread(path)
    image = np.array(np.float32(image))

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
    results = SegAutoMaskPredictor().image_predict(
        source="images/3/img_3_20.png",
        model_type="vit_l", # vit_l, vit_h, vit_b
        points_per_side=16,
        points_per_batch=64,
        min_area=0,
        output_path="images/3/img_3_20_output.png",
        # show=True,
        save=False,
    )
    features_other = apply_solider_v2('images/3/img_3_20.png')
    features_other2 = apply_solider_v2('images/3/img_3_60.png')
    distance = euclidean_distance(features_other,features_other2)
    print(distance)
    # join_images = np.stack([features_other,features_other2])
    # k = 2
    # kmodel = KMeans(n_clusters=k, random_state=728, n_init=10)
    # kmodel.fit(join_images)
    # kpredictions = kmodel.predict(join_images)
    # TEST WITH OTHER IMAGE


compare_2_images()

