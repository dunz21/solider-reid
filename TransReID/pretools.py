from config import cfg
from .model import make_model
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


def load_model():
    cfg.merge_from_file("./TransReID/configs/market/vit_transreid.yml")
    # cfg.merge_from_file("./configs/market/vit_transreid_stride.yml")
    # cfg.MODEL.PRETRAIN_PATH =  "TransReID/model/jx_vit_base_p16_224-80ecf9dd.pth"
    # cfg.TEST.WEIGHT =  "TransReID/model/vit_transreid_market.pth"
    
    # cfg.TEST.WEIGHT =  './model/swin_base_market.pth'

    model = make_model(cfg, num_class=0, camera_num=1, view_num = 6)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    assert cfg.TEST.WEIGHT != ''
    return model


# If the image is torch Tensor, it is expected to have […, H, W] shape
def transform_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(np.float32(image))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256,128), antialias=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image)
    return image

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img