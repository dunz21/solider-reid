from utils.colab import solider_result,plot_pca,plot_tsne,plot_mds,plot_nmf,plot_svd

if __name__ == "__main__":
    folder_path = './images_subframev2/1'
    test = [
        # './images_subframev2/90',
        # './images_subframev2/10',
        # './images_subframev2/44',
        # './images_subframev2/72',
        # './images_subframev2/78',
        './images_subframev2/1',
        # './images_subframev2/2',
        './images_testsss',
        ]
    features , images_names = solider_result(folder_path=test, weight='./model/swin_base_market.pth')
    plot_pca(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_tsne(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_mds(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    # plot_svd(features_array=features, image_names=images_names,simpleLegend=True, title='TEST swin_base_market')
    