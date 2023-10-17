from utils.heatmap import *

if __name__ == "__main__":
    heatmap_solider('people_2',weight='./model/swin_base_market.pth',semantic_weight=0.2)
    heatmap_transreid('people_2',pretrain_path="TransReID/model/jx_vit_base_p16_224-80ecf9dd.pth",weight="TransReID/model/vit_transreid_market.pth")
    heatmap_alignreid('people_2',weight="Alignedreid/Cuhk03_Resnet50_Alignedreid/checkpoint_ep300.pth.tar")