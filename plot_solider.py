from utils.colab import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector

def final():
    features_from_csv = pd.read_csv('solider.csv').sort_values(by='folder')
    count_folders_items = features_from_csv.groupby('folder').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    filter = count_folders_items[count_folders_items.Count > 19]
    total_folder_to_traverse =filter['folder'].values

    for i in range(len(total_folder_to_traverse)):
        for j in range(i + 1, len(total_folder_to_traverse)):
            folderA = total_folder_to_traverse[i] 
            folderB = total_folder_to_traverse[j]
            valuesA = features_from_csv[features_from_csv.folder == folderA]
            valuesB = features_from_csv[features_from_csv.folder == folderB]
            totalValues = pd.concat([valuesA,valuesB],ignore_index=True)
            images_names , features = totalValues.iloc[:, 1], totalValues.iloc[:, 2:].values
            match, msg = plot_mds_dbscan(features_array=features, image_names=images_names,simpleLegend=True, title='DB',eps=9,min_samples=14)
            # if(match):
            #     print(f"Calculation between {folderA} and {folderB}: {msg}")



def test():
    test = [
    # './images_subframev2',
    './images_subframev2/41',
    # './images_subframev2/2',
    # './images_subframev2/114',
    # './images_subframev2/90',
    # './images_subframev2/130',
    # './images_subframev2/139',
    # './images_subframev2/89',
    # './images_subframev2/6',
    # './images_subframev2/45',
    # './images_subframev2/108',
    # './images_subframev2/158',
    # './images_subframev2/10',
    # './images_subframev2/44',
    # './images_subframev2/72',
    # './images_subframev2/2',
    # './images_subframev2/90',
    # './images_subframev2/77',
    # './images_subframev2/130',
    # './images_subframev2/2',
    # './images_testsss',
    ]
    models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
    ]


    backends = [
        'opencv', 
        'ssd', 
        'dlib', 
        'mtcnn', 
        'retinaface', 
        'mediapipe',
        'yolov8',
        'yunet',
        ]
    
    
    result = face_id_details(folder_path=test,model=models[2],backend=backends[4])
    return False
    # features , images_names = solider_result(folder_path=test, weight='./model/swin_base_market.pth')
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[0])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[1])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[2])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[3])
    # features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[0])
    features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[1])
    # features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[2])
    features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[3])
    features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[4])
    # features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[5])
    # features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[6])
    # features_2 , images, images_names_2, facial_area, confidence = face_id_results(folder_path=test,model=models[2],backend=backends[7])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[5])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[6])
    # features_2 , images_names_2 = face_id_results(folder_path='people_faces',model=models[2],backend=backends[7])
    # features_3 , images_names_3 = alignedreid_result(folder_path=test, weight='./model/swin_base_market.pth')

    # plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='solider_result',scaler=True)
    # plot_mds(features_array=features_2, image_names=images_names_2,simpleLegend=True, title='transreid_result')
    # plot_mds(features_array=features_3, image_names=images_names_3,simpleLegend=True, title='alignedreid_result',scaler=True)


if __name__ == "__main__":
    test()
    exit()
    features_from_csv = pd.read_csv('solider.csv').sort_values(by='folder')
    A=16
    B=89
    images_names, features = features_from_csv[(features_from_csv.folder == A) | (features_from_csv.folder == B)].iloc[:, 1].values, features_from_csv[(features_from_csv.folder == A) | (features_from_csv.folder == B)].iloc[:, 2:].values
    # plot_mds_dbscan(features_array=features, image_names=images_names,plot=True, title='DB',eps=9,min_samples_ratio=0.15,min_include=3)
    # plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_mds_kmeans(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market',scaler=False,n_clusters=3)
    # plot_mds_gmm(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market',scaler=False,n_clusters=3)
    # final()
    # exit()
    folder_path = './images_subframev2/1'
    test = [
        # './images_subframev2',
        './images_subframev2/90',
        './images_subframev2/2',
        # './images_subframev2/114',
        # './images_subframev2/90',
        # './images_subframev2/130',
        # './images_subframev2/139',
        # './images_subframev2/89',
        # './images_subframev2/6',
        # './images_subframev2/45',
        # './images_subframev2/108',
        # './images_subframev2/158',
        # './images_subframev2/10',
        # './images_subframev2/44',
        # './images_subframev2/72',
        # './images_subframev2/2',
        # './images_subframev2/90',
        # './images_subframev2/77',
        # './images_subframev2/130',
        # './images_subframev2/2',
        # './images_testsss',
        ]
    features , images_names = solider_result(folder_path=test, weight='./model/swin_base_market.pth')
    features_2 , images_names_2 = transreid_result(folder_path=test, pretrain_path="TransReID/model/jx_vit_base_p16_224-80ecf9dd.pth",weight="TransReID/model/vit_transreid_market.pth")
    features_3 , images_names_3 = alignedreid_result(folder_path=test, weight='./model/swin_base_market.pth')

    # df = pd.DataFrame(features)
    # names = pd.DataFrame({'folder':[img.split('_')[1] for img in images_names],'images': images_names})
    # result = pd.concat([names, df], axis=1)
    # result.to_csv('solider_2_90.csv',index=False)

    # features_from_csv = pd.read_csv('solider_2_90.csv').sort_values(by='folder')
    # images_names, features = features_from_csv.iloc[:, 1], features_from_csv.iloc[:, 2:].values


    # print('asd')

    # compute_distance_matrix(features,images_names,['img_1_820','img_1_940','img_1_570','img_1_980'])
    # plot_distance_heatmap(features,images_names,distance_type='cosine')
    # plot_pca(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_tsne(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market',perplexity=500)
    # plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_mds_dbscan(features_array=features, image_names=images_names,simpleLegend=True, title='DB',eps=9,min_samples=14)
    # plot_svd(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')

    # df = create_dataframe_from_folder('./images_subframev2')
    # df = df.groupby('Folder').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    # print(df)