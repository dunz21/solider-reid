from utils.colab import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.pyplot as plt
from torchinfo import summary
from utils.pretools import load_model as load_model_solider
import numpy as np
from sklearn.metrics import silhouette_score, euclidean_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import re


def evaluate_model(features, image_names):
    """
    Evaluate the model based on the features and image names provided.
    
    Args:
    features (np.array): The array of features with shape (num_images, feat_vector).
    image_names (list): The list of image names with labels.
    
    Returns:
    dict: A dictionary with the calculated metrics and a plot of MDS.
    """
    
    # Extract the IDs from image names using regex
    ids = [int(re.search(r'img_(\d+)_', name).group(1)) for name in image_names]
    
    # Calculate pairwise Euclidean distances
    pairwise_dist = euclidean_distances(features)
    
    # Compute intra-class and inter-class distances
    intra_class_dists = []
    inter_class_dists = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if ids[i] == ids[j]:
                intra_class_dists.append(pairwise_dist[i, j])
            else:
                inter_class_dists.append(pairwise_dist[i, j])
    
    # Calculate the average intra-class and inter-class distance
    avg_intra_class_dist = np.mean(intra_class_dists)
    avg_inter_class_dist = np.mean(inter_class_dists)
    
    # Compute the silhouette score
    silhouette_avg = silhouette_score(features, ids)
    
    # Perform MDS and plot
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(pairwise_dist)
    
    # Prepare the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], c=ids, cmap='Spectral')
    plt.title('MDS Projection of Features')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.colorbar(scatter, label='Person IDs')
    plt.show()
    
    # Return the computed metrics
    return {
        'avg_intra_class_dist': avg_intra_class_dist,
        'avg_inter_class_dist': avg_inter_class_dist,
        'silhouette_score': silhouette_avg
    }



if __name__ == "__main__":
    test = [
        '/home/diego/Downloads/DataAdicionalTest/1667',
      #  '/home/diego/Downloads/DataAdicionalTest/2836',
     #   '/home/diego/Downloads/DataAdicionalTest/3850',
        ]

    features , images_names = solider_result(folder_path=test, weight='./log/market1501/swin_base20/transformer_100.pth')
    #plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='transformer_100-20SW')
    result = evaluate_model(features=features,image_names=images_names)
    print(result)



