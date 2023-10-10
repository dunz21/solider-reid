from utils.colab import plot_pca

if __name__ == "__main__":
    folder_path = './images_subframev2/1'
    test = ['./images_subframev2/1','./images_subframev2/2','./images_subframev2/3']
    plot_pca(folder_path=test,simpleLegend=True,title='TEST XX',weight='./model/swin_base_market.pth')