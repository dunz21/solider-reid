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

def plot_pca(folder_path="",simpleLegend=True, title=""):
    model = solider_model('./model/swin_base_market.pth')
    images = glob.glob(os.path.join(folder_path, '*.png'))
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