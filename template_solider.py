from config import cfg
from model import make_model
import cv2
from torchvision import transforms as pth_transforms
import numpy as np

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