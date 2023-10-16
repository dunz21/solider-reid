import torch
from config import cfg
from model import make_model
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


def load_model():
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.TEST.WEIGHT =  './model/swin_base_market.pth'

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    assert cfg.TEST.WEIGHT != ''
    return model

# If the image is torch Tensor, it is expected to have […, H, W] shape
def preprocess_image(img_path, heigth,width):
    transform = transforms.Compose([
        transforms.Resize((heigth, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    return image



def transform_image(path,version=1):
    image = cv2.imread(path)
    if version == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif version == 2:
        image = np.array(np.float32(image))
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(np.float32(image))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((384,128), antialias=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image)
    return image