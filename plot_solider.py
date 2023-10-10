from config import cfg
from model import make_model
import torch
import cv2
from torchvision import transforms as pth_transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os
import glob
from utils.colab import preprocess_image,solider_model
import seaborn as sns

def plot_pca(images, model, preprocess_image,simpleLegend=True, title=""):
    """
    Plots the output of a model on images in 2D using PCA.
    
    Parameters:
    - images: List of images.
    - model: A model that will extract features.
    - preprocess_image: A function to preprocess images before passing to the model.
    """
    
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    features_list = []
    for img in images:
        img = preprocess_image(img)
        with torch.no_grad():
            features, _ = model(img)
        features_list.append(features)
    
    # Concatenate features using torch.cat
    features_tensor = torch.cat(features_list, dim=0)
    
    # Convert tensor to numpy array
    features_array = features_tensor.cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_array)
    
    # Extract prefix and suffix from image names for coloring
    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    # Create a mapping of prefix to palette index
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    # Used to track which prefixes have been added to the legend already
    added_to_legend = set()
    legend_handles_labels = []

    # Plotting
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(pca_result):
        color = palette[prefix_to_index[prefixes[i]]]
        label = None
        if simpleLegend:
            if prefixes[i] not in added_to_legend:
                label = f"img_{prefixes[i]}"
                added_to_legend.add(prefixes[i])
        else:
            label = f"img_{prefixes[i]}_{suffixes[i]}"
            added_to_legend.add(prefixes[i])
        plt.text(x, y, f"{prefixes[i]}_{suffixes[i]}", fontsize=8, ha='right', va='bottom')
        handle = plt.scatter(x, y, color=(color[0], color[1], color[2], normalized_suffixes[i]), label=label)
        if label:
            legend_handles_labels.append((handle, label))

    # Sort the handles and labels
    legend_handles_labels = sorted(legend_handles_labels, key=lambda x: x[1])

    # Extract sorted handles and labels
    sorted_handles, sorted_labels = zip(*legend_handles_labels)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f"PCA Solider {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

if __name__ == "__main__":
    model = solider_model('./model/swin_base_market.pth')
    folder_path = './people_2'
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    plot_pca(image_files,model,preprocess_image,False,'TEST XX')