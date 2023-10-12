from utils.colab import solider_result,plot_pca,plot_tsne,plot_mds,plot_nmf,plot_svd,plot_distance_heatmap,compute_distance_matrix,create_dataframe_from_folder,plot_mds_dbscan
import pandas as pd
if __name__ == "__main__":
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
    # features , images_names = solider_result(folder_path=test, weight='./model/swin_base_market.pth')

    # df = pd.DataFrame(features)
    # names = pd.DataFrame({'folder':[img.split('_')[1] for img in images_names],'images': images_names})
    # result = pd.concat([names, df], axis=1)
    # result.to_csv('solider_2_90.csv',index=False)

    features_from_csv = pd.read_csv('solider_2_90.csv').sort_values(by='folder')
    images_names, features = features_from_csv.iloc[:, 1], features_from_csv.iloc[:, 2:].values


    # print('asd')

    # compute_distance_matrix(features,images_names,['img_1_820','img_1_940','img_1_570','img_1_980'])
    # plot_distance_heatmap(features,images_names,distance_type='cosine')
    # plot_pca(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_tsne(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market',perplexity=500)
    plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_mds_dbscan(features_array=features, image_names=images_names,simpleLegend=True, title='DB',eps=9,min_samples=14)
    # plot_svd(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')

    # df = create_dataframe_from_folder('./images_subframev2')
    # df = df.groupby('Folder').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    # print(df)