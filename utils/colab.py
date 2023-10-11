from config import cfg
from model import make_model
from torchvision import transforms as pth_transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import glob
import seaborn as sns
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD  # Importing TruncatedSVD
from scipy.spatial.distance import cdist


def solider_model(path):
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.TEST.WEIGHT =  path
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    assert cfg.TEST.WEIGHT != ''
    return model

def preprocess_image(img_path):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((384, 128)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)

def extract_images_from_subfolders(folder_paths):
    # If the input is a string (single folder path), convert it into a list
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    all_images = []
    
    for folder_path in folder_paths:
        # Walk through each main folder and its subfolders
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # For each subfolder, find all .png images
            images = glob.glob(os.path.join(dirpath, '*.png'))
            all_images.extend(images)
    return all_images

def solider_result(folder_path="", weight=''):
    model = solider_model(weight)
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    total_batch = [preprocess_image(img) for img in images]
    with torch.no_grad():
        features_list, _ = model(torch.cat(total_batch,dim=0))
    
    # Convert tensor to numpy array
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def plot_pca(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10)):
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_array)

    # Apply DBSCAN clustering
    # dbscan = DBSCAN(eps=4, min_samples=4)  # Adjust these parameters as needed
    # cluster_labels = dbscan.fit_predict(pca_result)
    # cluster_palette = sns.color_palette("husl", len(set(cluster_labels)))

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
    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(pca_result):
        color = palette[prefix_to_index[prefixes[i]]]
        # cluster_color = cluster_palette[cluster_labels[i]]

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
        # handle = plt.scatter(x, y, color=cluster_color, label=label)

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

def plot_tsne(features_array="", image_names=[],simpleLegend=True, title="",perplexity=5, figsize=(12,10)):
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300)
    tsne_result = tsne.fit_transform(features_array)
    
    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    added_to_legend = set()
    legend_handles_labels = []

    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(tsne_result):
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

    legend_handles_labels = sorted(legend_handles_labels, key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*legend_handles_labels)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f"t-SNE Solider {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def plot_mds(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10)):
    # Apply MDS
    mds = MDS(n_components=2)
    mds_result = mds.fit_transform(features_array)
    
    # Extract prefix and suffix from image names for coloring
    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    # Create a mapping of prefix to palette index
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    added_to_legend = set()
    legend_handles_labels = []

    # Plotting
    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(mds_result):
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
    sorted_handles, sorted_labels = zip(*legend_handles_labels)

    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title(f"MDS Solider {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def plot_nmf(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10)):
    # Apply NMF
    nmf = NMF(n_components=2)
    nmf_result = nmf.fit_transform(features_array)
    
    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    added_to_legend = set()
    legend_handles_labels = []

    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(nmf_result):
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

    legend_handles_labels = sorted(legend_handles_labels, key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*legend_handles_labels)

    plt.xlabel('NMF Component 1')
    plt.ylabel('NMF Component 2')
    plt.title(f"NMF Solider {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def plot_svd(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10)):
    # Apply SVD
    svd = TruncatedSVD(n_components=2)  # Using TruncatedSVD for dimensionality reduction
    svd_result = svd.fit_transform(features_array)

    prefixes = [int(name.split('_')[1]) for name in image_names]
    suffixes = [int(name.split('_')[2]) for name in image_names]
    max_suffix = max(suffixes)
    min_alpha = 0.6
    normalized_suffixes = [min_alpha + (1 - min_alpha) * (s / max_suffix) for s in suffixes]
    
    unique_prefixes = list(set(prefixes))
    prefix_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes)}
    added_to_legend = set()
    legend_handles_labels = []
    
    # Plotting
    plt.figure(figsize=figsize)
    palette = sns.color_palette("husl", len(unique_prefixes))
    for i, (x, y) in enumerate(svd_result):  # Change pca_result to svd_result
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

    legend_handles_labels = sorted(legend_handles_labels, key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*legend_handles_labels)
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f"SVD Solider {title}")  # Change PCA to SVD in the title
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def plot_distance_heatmap(features, image_names, distance_type='euclidean'):
    """
    Plots a heatmap matrix of pairwise distances between images.

    Parameters:
    - features: array-like of shape [X,1024] representing the features of each image.
    - image_names: list of image names.
    - distance_type: 'euclidean' or 'cosine' for the type of distance to compute.
    """
    if distance_type == 'euclidean':
        distances = cdist(features, features, 'euclidean')
    elif distance_type == 'cosine':
        distances = cdist(features, features, 'cosine')
    else:
        raise ValueError("Invalid distance_type. Choose 'euclidean' or 'cosine'.")

    plt.figure(figsize=(10, 8))
    plt.imshow(distances, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'{distance_type.capitalize()} Distance Heatmap')
    plt.xticks(np.arange(len(image_names)), image_names, rotation=90)
    plt.yticks(np.arange(len(image_names)), image_names)
    plt.tight_layout()
    plt.show()

def compute_distance_matrix(features, image_names, selected_names):
    """
    Compute pairwise distance matrix for selected image names.
    
    Parameters:
    - features (np.array): Array of shape [X, 1024] containing the features for each image.
    - image_names (list): List of image names corresponding to the features.
    - selected_names (list): List of selected image names for which the distance matrix should be computed.
    
    Returns:
    - distance_matrix (np.array): Pairwise distance matrix for the selected image names.
    """

    # Extract features for selected images
    selected_features = [features[image_names.index(name)] for name in selected_names]
    
    # Compute pairwise distances
    distance_matrix = np.zeros((len(selected_names), len(selected_names)))
    for i in range(len(selected_names)):
        for j in range(len(selected_names)):
            distance_matrix[i, j] = np.linalg.norm(selected_features[i] - selected_features[j])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, xticklabels=selected_names, yticklabels=selected_names, annot=True, cmap='YlGnBu', cbar_kws={"label": "Distance"})
    plt.title('Pairwise Distance Heatmap')
    plt.show()
    return distance_matrix
