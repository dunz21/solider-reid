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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances,pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.pretools import load_model as load_model_solider,preprocess_image
from TransReID.pretools import load_model as load_model_transreid
from AlignedReID.pretools import load_model as load_model_alignreid

#Probar normalizar datos. Aplicar clustering antes!!! Que puta es el 1024, y aplicar medidas locales
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

#### MODELS

def solider_result(folder_path="", weight='',semantic_weight=0.2):
    model = load_model_solider(weight,semantic_weight)
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,384,128)], dim=0) for img in images]
    with torch.no_grad():
        features_list, _ = model(torch.cat(total_batch,dim=0))
    
    # Convert tensor to numpy array
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def transreid_result(folder_path="",pretrain_path="",weight=""):
    model = load_model_transreid(pretrain_path=pretrain_path,weight=weight)
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,256,128)], dim=0) for img in images]
    with torch.no_grad():
        features_list = model(torch.cat(total_batch,dim=0), cam_label=0, view_label=0)
    
    # Convert tensor to numpy array
    features_array = features_list.cpu().numpy()
    return features_array, image_names

def alignedreid_result(folder_path="", weight=''):
    # model_path = "AlignedReID/Cuhk03_Resnet50_Alignedreid/checkpoint_ep300.pth.tar" #FUNCIONA
    model = load_model_alignreid(model_path=weight)
    images = extract_images_from_subfolders(folder_path)
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    total_batch = [torch.stack([preprocess_image(img,384,128)], dim=0) for img in images]
    with torch.no_grad():
        features_list, _ = model(torch.cat(total_batch,dim=0))
    
    # Convert tensor to numpy array
    features_array = features_list.cpu().numpy()
    return features_array, image_names

#### MODELS

def heatmap_solider(folder_path,weight,semantic_weight=0.2):
    model = load_model_solider(weight=weight,semantic_weight=semantic_weight)
    images = extract_images_from_subfolders(folder_path)
    images = sorted(images)

    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    total_batch = [preprocess_image(img ,384,128) for img in images]
    with torch.no_grad():
        list_features,_ = model(torch.stack(total_batch, dim=0))

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distances(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity SOLIDER')
    plt.show()

def heatmap_transreid(folder_path,pretrain_path="TransReID/model/jx_vit_base_p16_224-80ecf9dd.pth",weight="TransReID/model/vit_transreid_market.pth"):
    model = load_model_transreid(pretrain_path=pretrain_path,weight=weight)
    images = extract_images_from_subfolders(folder_path)
    images = sorted(images)
    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    total_batch = [preprocess_image(img ,256,128) for img in images]
    with torch.no_grad():
        list_features = model(torch.stack(total_batch, dim=0), cam_label=0, view_label=0)

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distances(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity TransReID')
    plt.show()

def heatmap_alignreid(folder_path,weight):
    model = load_model_alignreid(model_path=weight)
    images = extract_images_from_subfolders(folder_path)
    images = sorted(images)
    # model_path = "" #FUNCIONA
    # model_path = "AlignedReID/Market1501_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar" #FUNCIOAN
    results_matrix = []
    
    # Version with all images and 1 infer in a total array (IMPROVE PERFORMANCE)
    total_batch = [preprocess_image(img ,384,128) for img in images]
    with torch.no_grad():
        list_features,_ = model(torch.stack(total_batch, dim=0))

    for feat in list_features:
        img_results = []
        for feat2 in list_features:
            result_difference = euclidean_distances(torch.stack([feat2],dim=0), torch.stack([feat],dim=0))
            img_results.append(result_difference[0][0].item())  # Store as float
        results_matrix.append(img_results)

    # Create a DataFrame with the results
    df = pd.DataFrame(results_matrix, columns=images, index=images)
    
    # Write the DataFrame to a CSV file
    # df.to_csv('similar_images_result_'+folder_name+'.csv')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="RdYlGn_r", fmt=".2f")  # Red-Yellow-Green reversed colormap
    plt.title('Similarity AlignReID')
    plt.show()





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
    plt.title(f"PCA {title}")
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
    plt.title(f"t-SNE({perplexity}) {title}")
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def plot_mds_dbscan(features_array="", image_names=[], plot=True, title="", figsize=(12, 10), eps=0.5, min_samples_ratio=0.15, min_include=3, scaler=True):
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(features_array)
    
    count_image_cluster = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names]}).groupby('id').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    idcluster1 , sizecluster1 = count_image_cluster.iloc[0,0] , count_image_cluster.iloc[0,1]
    idcluster2 , sizecluster2 = count_image_cluster.iloc[1,0] , count_image_cluster.iloc[1,1]
    

    # Apply DBSCAN clustering
    min_samples = int(sizecluster1*min_samples_ratio)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(mds_result)
    labels = db.labels_

    
    ### DATA FRAME ####
    data_images = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names],'labels': db.labels_})
    count_data = data_images[data_images.labels != -1].groupby('labels').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    if len(count_data) == 0:
        return False, ''
    id_biggest_cluster_size = count_data.iloc[0,0]
    overlap_images = data_images[data_images.labels == id_biggest_cluster_size].groupby('id').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    if len(overlap_images) > 1:
        if overlap_images.iloc[1,1] > min_include:
            total_images_inside_big_cluster = ', '.join([f"ID: {row[0]} Total: {row[1]}" for index,row in overlap_images.iloc[1:].reset_index(drop=True).iterrows()])
            msg = f"Total de imagenes {total_images_inside_big_cluster} encontradas en cluster ID: {idcluster1}  min_samples: {min_samples}"
            print(msg)
    ### DATA FRAME ####

    if plot:
        # Define a color palette for DBSCAN clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # excluding noise
        cluster_palette = sns.color_palette('husl', n_clusters)
        # Handle noise in the data
        colors = [(0.5, 0.5, 0.5) if label == -1 else cluster_palette[label] for label in labels]
        
        # Plotting
        plt.figure(figsize=figsize)
        for i, (x, y) in enumerate(mds_result):
            plt.text(x, y, f"{image_names[i]}", fontsize=8, ha='right', va='bottom')
            plt.scatter(x, y, color=colors[i], label=f'Cluster {labels[i]}' if labels[i] != -1 else 'Noise')

        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.title(f"MDS/ DBSCAN eps {eps} min_samples {min_samples} {title}")
        
        # Create a legend for the clusters
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_palette[i], markersize=10) for i in range(n_clusters)]
        labels = [f'Cluster {i}' for i in range(n_clusters)]
        if -1 in labels:  # if there's noise
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5), markersize=10))
            labels.append('Noise')
        plt.legend(handles=handles, labels=labels)
        plt.show()

def plot_mds(features_array="", image_names=[],simpleLegend=True, title="", figsize=(12,10), scaler=True):
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
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
    plt.title(f"MDS {title}")
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
    plt.title(f"NMF {title}")
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
    plt.title(f"SVD {title}")  # Change PCA to SVD in the title
    plt.legend(handles=sorted_handles, labels=sorted_labels)
    plt.show()

def create_dataframe_from_folder(folder_path):
    image_list = []
    folder_list = []

    # Loop through each subfolder in the main folder
    for folder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder)

        # Ensure that the current item is a folder (directory)
        if os.path.isdir(subfolder_path):
            for image in os.listdir(subfolder_path):

                # Append the image name and folder name to respective lists
                image_list.append(image)
                folder_list.append(folder)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Images': image_list,
        'Folder': folder_list
    })

    return df

def plot_mds_gmm(features_array="", image_names=[], simpleLegend=True, title="", figsize=(12,10), scaler=True, n_clusters=3):
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
        
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(features_array)
    
    # Apply GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(mds_result)
    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(mds_result, gmm_labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    # Define a color palette for GMM clusters
    cluster_palette = sns.color_palette('husl', n_clusters)
    
    data_images = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names],'labels': gmm_labels})
    grouped = data_images.groupby(['labels', 'id']).size().reset_index(name='count')
    print(grouped)

    # Plotting
    plt.figure(figsize=figsize)
    for i, (x, y) in enumerate(mds_result):
        plt.text(x, y, f"{image_names[i]}", fontsize=8, ha='right', va='bottom')
        plt.scatter(x, y, color=cluster_palette[gmm_labels[i]], label=f'Cluster {gmm_labels[i]}' if simpleLegend else image_names[i])

    # Create a legend for the clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_palette[i], markersize=10) for i in range(n_clusters)]
    labels = [f'Cluster {i}' for i in range(n_clusters)]
    plt.legend(handles=handles, labels=labels)
    
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title(f"MDS with GMM clustering {title}")
    
    plt.show()

def plot_mds_kmeans(features_array="", image_names=[], simpleLegend=True, title="", figsize=(12,10), scaler=True, n_clusters=3):
    
    if scaler:
        scaler = StandardScaler().fit(features_array)
        features_array = scaler.transform(features_array)
        
    # Apply MDS
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(features_array)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(mds_result)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(mds_result, kmeans_labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    
    # Define a color palette for KMeans clusters
    cluster_palette = sns.color_palette('husl', n_clusters)
    

    ### DATA FRAME ####
    data_images = pd.DataFrame({'images': image_names,'id': [img.split('_')[1] for img in image_names],'labels': kmeans_labels})
    grouped = data_images.groupby(['labels', 'id']).size().reset_index(name='count')
    print(grouped)
    ### DATA FRAME ####








    # Plotting
    plt.figure(figsize=figsize)
    for i, (x, y) in enumerate(mds_result):
        plt.text(x, y, f"{image_names[i]}", fontsize=8, ha='right', va='bottom')
        plt.scatter(x, y, color=cluster_palette[kmeans_labels[i]], label=f'Cluster {kmeans_labels[i]}' if simpleLegend else image_names[i])

    # Create a legend for the clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_palette[i], markersize=10) for i in range(n_clusters)]
    labels = [f'Cluster {i}' for i in range(n_clusters)]
    plt.legend(handles=handles, labels=labels)
    
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title(f"MDS with KMeans clustering {title}")
    
    plt.show()