import os, shutil
from config import cfg
from datetime import datetime
from datasets import make_dataloader
import cv2
from PIL import Image
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
from utils.pretools import load_model,preprocess_image
from TransReID.pretools import load_model as load_model_transreid
from AlignedReID.pretools import load_model as load_model_alignreid


def similar_between_images(folder_name):
    image_folder = folder_name
    images = [filename for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
    images = sorted(images)
    model = load_model()
    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    img_transform = [preprocess_image(os.path.join(image_folder, img),384,128) for img in images]
    with torch.no_grad():
        list_features,_ = model(torch.stack(img_transform, dim=0), cam_label=0, view_label=0)

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distance(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity SOLIDER')
    plt.show()

def similar_between_images_transreid(folder_name):
    image_folder = folder_name
    images = [filename for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
    images = sorted(images)
    model = load_model_transreid()
    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    img_transform = [preprocess_image(os.path.join(image_folder, img),256,128) for img in images]
    with torch.no_grad():
        list_features = model(torch.stack(img_transform, dim=0), cam_label=0, view_label=0)

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distance(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity TransReID')
    plt.show()

def similar_between_images_alignreid(folder_name):
    image_folder = folder_name
    images = [filename for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
    images = sorted(images)
    model = load_model_alignreid()
    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    img_transform = [preprocess_image(os.path.join(image_folder, img),384,128) for img in images]
    with torch.no_grad():
        list_features,_ = model(torch.stack(img_transform, dim=0))

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distance(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity AlignReID')
    plt.show()

similar_between_images('people_2')
similar_between_images_transreid('people_2')
similar_between_images_alignreid('people_2')