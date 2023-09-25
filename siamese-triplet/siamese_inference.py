import torch
from torchvision import transforms
from PIL import Image
import os
from collections import namedtuple
import cv2
import torch.nn.functional as F
import torch.nn as nn
import embedding

class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        return E1, E2, E3

def get_model(args, device):
    embeddingNet = embedding.EmbeddingResnet()

    model = TripletNet(embeddingNet)
    model = nn.DataParallel(model, device_ids=args.gpu_devices)
    model = model.to(device)
    checkpoint = torch.load('siamese-triplet/checkpoint_5.pth')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def test_triplet_similarity(anchor_img_path, pos_img_path, neg_img_path, checkpoint_path, margin=1.0):
    if not os.path.exists(checkpoint_path) or not all(map(os.path.exists, [anchor_img_path, pos_img_path, neg_img_path])):
        print("Files do not exist")
        return

    Args = namedtuple('Args', ['dataset', 'gpu_devices', 'ckp'])
    # Create an instance of the named tuple
    args = Args(dataset='custom', gpu_devices=[0], ckp=checkpoint_path)
    model = get_model(args, 'cpu')

    
    # Load model checkpoint
    # model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    # Read and resize images with OpenCV
    anchor_img = cv2.imread(anchor_img_path)
    pos_img = cv2.imread(pos_img_path)
    neg_img = cv2.imread(neg_img_path)

    anchor_img = cv2.resize(anchor_img, (228, 228))
    pos_img = cv2.resize(pos_img, (228, 228))
    neg_img = cv2.resize(neg_img, (228, 228))

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Standard normalization
    ])
    criterion = torch.nn.MarginRankingLoss(margin=margin)
    # Apply transforms
    anchor_img = transform(anchor_img).unsqueeze(0)
    pos_img = transform(pos_img).unsqueeze(0)
    neg_img = transform(neg_img).unsqueeze(0)

    with torch.no_grad():
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        
        print(f'Test Loss: {loss.item()}')

        # Test similarity based on a threshold (here, it's the margin)
        prediction = (dist_E1_E3 - dist_E1_E2 - margin).cpu().data
        prediction = prediction.view(prediction.numel())
        
        # Convert the prediction to either 0 or 1
        prediction = (prediction > 0).float()
        
        # Calculate accuracy (it will be either 0 or 1 for a single triplet)
        batch_acc = prediction.sum() * 1.0 / prediction.numel()

        print(f'Test Accuracy (anchor closer to positive if accuracy is 1): {batch_acc}')

# Replace placeholders with your paths and classes
anchor_img_path = 'img_3_20.png'
pos_img_path = 'img_3_120.png'
neg_img_path = 'img_3_420.png'
checkpoint_path = './results_diego/Custom_exp1/checkpoint_50.pth'

# Test the triplet similarity
test_triplet_similarity(anchor_img_path, pos_img_path, neg_img_path, checkpoint_path)