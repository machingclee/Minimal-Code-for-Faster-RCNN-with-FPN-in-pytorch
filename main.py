import torch
import os
from src.train import train_with_nan
from src.faster_rcnn_renset_fpn import FasterRCNNResnet50FPN
from src.device import device
from src.train import TrainingErrorMessage
from src.visualize import inference
from PIL import Image
from tqdm import tqdm


def predict_on_trip():
    model_path = "pths/model_epoch_50.pth"

    faster_rcnn = FasterRCNNResnet50FPN().to(device)
    
    if model_path is not None:
        faster_rcnn.load_state_dict(torch.load(model_path))
        
    faster_rcnn.eval()
    
    # train_with_nan(
    #     faster_rcnn,
    #     lr=1e-5,
    #     start_epoch=1,
    #     epoches=50,
    #     save_weight_interval=1
    # )
    target_trip = "2022-3rd_TRIP-17" 
    targets_dir = "prediction_targets"
    results_dir = "prediction_results"
    
    if not os.path.exists(f"{results_dir}/{target_trip}"):
        os.makedirs(f"{results_dir}/{target_trip}/normal")
        os.makedirs(f"{results_dir}/{target_trip}/rust")
                    
    for img_basename in tqdm(os.listdir(f"{targets_dir}/{target_trip}")):
        img_path = f"{targets_dir}/{target_trip}/{img_basename}"        
        img = Image.open(img_path)
        score = inference(faster_rcnn, img)
        result_img_name = img_basename.replace(".jpg", "") + "_" + str(score) + ".jpg"
        
        if score > 0.4:   
            cat_dir = "rust" 
        else:
            cat_dir = "normal"
            
        result_path = f"{results_dir}/{target_trip}/{cat_dir}/{result_img_name}"
        img.save(result_path)




def main():
    predict_on_trip()


if __name__ == "__main__":
    main()
