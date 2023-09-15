import os, shutil
from config import cfg
from datasets import make_dataloader
from model import make_model
import cv2
import time
import torch
from torchvision import transforms as pth_transforms
from ultralytics import YOLO
from utils.metrics import euclidean_distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EUCLEDIAN_DISTANCE = 150 #Valor máximo de la distancia euclidea para considerar misma persona. Tiene que configurarse de forma empírica
def solider_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Selecciona GPU si está disponible
    cfg.merge_from_file("./configs/market/swin_base.yml")
    cfg.MODEL.SEMANTIC_WEIGHT =  0.2
    cfg.TEST.WEIGHT =  './model/swin_base_market.pth'

    #Crea el modelo y carga los pesos
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    model.eval().to(device)
    return model

def save_images_based_on_id(n_frame,sub_frame,id):
    if n_frame == 1:
        if os.path.exists('images'):
            shutil.rmtree('images')
    if n_frame % 20 == 0:
        
        # Define the directory for this ID
        id_directory = os.path.join('images', str(id))
        
        # Create the directory if it doesn't exist
        if not os.path.exists(id_directory):
            os.makedirs(id_directory)
        
        # Define the save path for the image within the ID-specific directory
        save_path = os.path.join(id_directory, f"img_{id}_{n_frame}.png")
        
        # Save the image
        cv2.imwrite(save_path, sub_frame)

def plot_feature_maps(feature_maps):
    # Convert tensors to NumPy arrays and reshape
    feature_maps_np = [feature_map.squeeze().cpu().numpy() for feature_map in feature_maps]
    num_feature_maps = len(feature_maps_np)
    num_visualization_channels = 3
    num_rows = num_visualization_channels
    num_cols = num_feature_maps

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))  # Adjust figsize as needed

    for i in range(num_feature_maps):
        for j in range(num_visualization_channels):
            ax = axes[j, i]
            sns.heatmap(feature_maps_np[i][j], ax=ax)
            ax.set_title(f"Feature Map {i + 1}, Channel {j + 1}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_feat_heatmap(feat):
    # Convert the feat tensor to a NumPy array
    feat_np = feat.squeeze().cpu().numpy()

    # Reshape feat_np to match the heatmap requirements
    feat_np_reshaped = feat_np.reshape(1, -1)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(feat_np_reshaped, cmap='viridis', annot=False)
    plt.title('Feature Map Heatmap')
    plt.xlabel('Feature')
    plt.ylabel('Channel')
    plt.show()

def track_by_feat(original_tracker_id,ids_feat_dict,feat):
    found = False
    found_tracks = ids_feat_dict.keys()
    #Si el track ya existe, mantengo el mismo número
    if original_tracker_id in found_tracks:
        tracker_feat_id = original_tracker_id
        found = True
    else:#si no existe, recorro cada uno de los ids almacenados y compruebo si hay similitud con alguno de ellos
        for k in found_tracks:
            distance = euclidean_distance(ids_feat_dict[k],feat)
            if distance < MAX_EUCLEDIAN_DISTANCE: #Si hay similitud, asigno el nuevo track al track de referencia
                print(f"Track {original_tracker_id} concuerda con ref {k}. Distancia euclidea: {distance}")
                tracker_feat_id = original_tracker_id
                found = True
                continue
    if not found: #Si el embedding no se corresponde con ninguno de los acumulados, asumimos que es una nueva persona
        print(f"No se encuentra feat para id {original_tracker_id}")
        ids_feat_dict[original_tracker_id] = feat
        tracker_feat_id = original_tracker_id
    return tracker_feat_id

def transform_image(image):
    image = np.array(np.float32(image))
    transform = pth_transforms.Compose(
        [
            pth_transforms.ToTensor(),
            pth_transforms.Resize((384,128)),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image)
    return image


def main():
    #Configuración de OpenCV
    cap = cv2.VideoCapture("./retail.mp4") #INTRODUCIR PATH DEL VIDEO.

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit

    #Configuración de archivo de guardado
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use appropriate codec (e.g., 'XVID', 'MJPG', 'H264', etc.)
    out = cv2.VideoWriter("/Users/diegosepulveda/Documents/diego/dev/ML/Cams/papers/SOLIDER-REID/output.mp4", fourcc, fps, (width, height)) #INTRODUCIR PATH DE GUARDADO

    
    yolo = YOLO("yolov8n.pt") #Crea modelo de detección
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Selecciona GPU si está disponible
    model = solider_model()
    t1=0
    t2=0
    t3=0
    t4=0
    #Inicialización de variables
    n_frame = 0
    ids_feat_dict = dict() #Guarda el embedding de cada detección. Si el embedding es similar al de referencia, se considera la misma persona y se asigna el mismo ID. Si no concuerda con ningún id de referencia se añade y se considera una persona nueva
    drawing_color = [0,0,255] #Color RGB de las detecciones
    while True:# True:
        n_frame +=1
        # Read a frame from the video
        ret, frame = cap.read()
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        detections = yolo.track(frame,verbose=False,persist=True,classes=0)[0] #Genera detecciones de personas
        boxes = detections.cpu().numpy().boxes.data
        for output in boxes: #Procesa cada persona por separado
            bbox_tl_x = int(output[0])
            bbox_tl_y = int(output[1])
            bbox_br_x = int(output[2])
            bbox_br_y = int(output[3])
            tracker_id = int(output[4])
            class_ = int(output[6])
            score = int(output[5])
            sub_frame = frame[bbox_tl_y:bbox_br_y,bbox_tl_x:bbox_br_x] #Extrae el sub frame donde aparece cada persona
            # save_images_based_on_id(n_frame,sub_frame,tracker_id)
            with torch.no_grad():
                s_frame = transform_image(sub_frame) #Aplica preprocesamiento a la imagen
                feat,feature_maps = model(torch.stack([s_frame], dim=0).to(device),cam_label=0, view_label=0) #Genera embedding
            # plot_feature_maps(feature_maps)
            # visualize_feat_heatmap(feat)
            tracker_feat_id = track_by_feat(tracker_id,ids_feat_dict,feat)
            cv2.rectangle(frame, (bbox_tl_x, bbox_tl_y),(bbox_br_x, bbox_br_y), color=drawing_color, thickness=2) #Draw detection rectangle
            cv2.putText(frame, f"{tracker_id} || {tracker_feat_id}", (bbox_tl_x, bbox_tl_y), cv2.FONT_HERSHEY_COMPLEX, 1, color=drawing_color, thickness=1) #Draw detection value

        out.write(frame) #Guarda frame
        cv2.imshow('/Users/diegosepulveda/Documents/diego/dev/ML/Cams/papers/SOLIDER-REID/retail.mp4',frame)
        print('frame {}/{} ({:.2f} ms) Tracker {:.1f} Tracker '.format(n_frame, int(frame_count),(t2-t1) * 1000,1E3 * (t4-t3)))
        # plt.pause(0.0001)  # Allow time for the event loop to update
        # cv2.waitKey(0) # DEBUG
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            break  # Exit the loop when 'Esc' (27) or 'q' is pressed
    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    


# similar_between_images()
main()
