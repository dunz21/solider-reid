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

def preprocess_image(img_path):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 128)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)

def plot_pca(images, model, preprocess_image):
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
    
    # Plotting
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(pca_result):
        plt.scatter(x, y, label=image_names[i])
    
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Model Solider')
    plt.show()

if __name__ == "__main__":
    model = solider_model()
    folder_path = './people_2'
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    plot_pca(image_files,model,preprocess_image)