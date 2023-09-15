import os, shutil
from config import cfg
from datasets import make_dataloader
from model import make_model
import cv2
import time
import torch
from torchvision import transforms as pth_transforms
from ultralytics import YOLO
from utils.metrics import euclidean_distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def solider_model():
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.TEST.WEIGHT =  './model/swin_base_market.pth'

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    assert cfg.TEST.WEIGHT != ''
    return model

def transform_image(path):
    image = cv2.imread(path)
    image = np.array(np.float32(image))
    # image = normalize(np.array(image).reshape(-1,3), axis=0)
    # image = image.reshape(384,128,3)
    # image = torch.from_numpy(image)

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Resize((384,128), antialias=True),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image)
    return image


def similar_between_images(folder_name):
    image_folder = folder_name
    images = [filename for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
    images = sorted(images)
    model = solider_model()
    results_matrix = []
    
    # Version with all images and 1 infere in a total array (IMPROVE PERFORMANCE)
    img_transform = [transform_image(os.path.join(image_folder, img)) for img in images]
    with torch.no_grad():
        list_features,_ = model(torch.stack(img_transform, dim=0), cam_label=0, view_label=0)

    
    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distance(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append("{:.2f}".format(result_difference[0][0]*1))
        results_matrix.append(img_results)


    # Iterative version one by one
    # with torch.no_grad():
    #     for img_name in images:
    #         img_path = os.path.join(image_folder, img_name)
    #         img = transform_image(img_path)
    #         img_result, _ = model(torch.stack([img], dim=0), cam_label=0, view_label=0)
            
    #         img_results = []
    #         for img_2_name in images:
    #             img_2_path = os.path.join(image_folder, img_2_name)
    #             img_2 = transform_image(img_2_path)
    #             img_2_result, _ = model(torch.stack([img_2], dim=0), cam_label=0, view_label=0)
    #             result_difference = euclidean_distance(img_2_result, img_result)
    #             img_results.append("{:.2f}".format(result_difference[0][0]*1))

    #         results_matrix.append(img_results)
    #         print(img_name)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    df.to_csv('similar_images_result_'+folder_name+'.csv')

similar_between_images('people3_bg')