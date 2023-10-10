from utils.colab import plot_pca

if __name__ == "__main__":
    folder_path = './people_2'
    plot_pca(folder_path=folder_path,simpleLegend=False,title='TEST XX',weight='./model/swin_base_market.pth')