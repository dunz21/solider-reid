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

def solider_model():
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.MODEL.TEST_WEIGHT =  './model/transformer_120.pth'
    cfg.merge_from_file("./configs/market/swin_base.yml")

    #Crea el modelo y carga los pesos
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    return model


def similar_between_images():
    image_folder = 'people'
    images = [filename for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
    model = solider_model()
    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    results_matrix = []

    with torch.no_grad():
        for img_name in images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            img_result, _ = model(torch.stack([transform(img)], dim=0), cam_label=0, view_label=0)
            
            img_results = []
            for img_2_name in images:
                img_2_path = os.path.join(image_folder, img_2_name)
                img_2 = cv2.imread(img_2_path)
                img_2_result, _ = model(torch.stack([transform(img_2)], dim=0), cam_label=0, view_label=0)
                result_difference = euclidean_distance(img_2_result, img_result)
                img_results.append(int(result_difference))

            results_matrix.append(img_results)
            print(img_name)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    df.to_csv('similar_images_result.csv')

similar_between_images()