import torch
from .model import init_model
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image


def load_model(model_path):
    model_alignedreid = init_model(name='resnet50', num_classes=0, loss={'softmax', 'metric'},aligned=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path,map_location=device)
    model_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in model_dict.items() if k not in ['classifier.weight', 'classifier.bias']}
    model_alignedreid.load_state_dict(pretrained_dict)
    model_alignedreid.to(device)
    model_alignedreid.eval()
    return model_alignedreid

# If the image is torch Tensor, it is expected to have […, H, W] shape
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((384, 128)),
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