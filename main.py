import torch
from src.train import train
from src.faster_rcnn_renset_fpn import FasterRCNNResnet50FPN
from src.device import device
from src.train import TrainingErrorMessage

def train_with_nan(
    faster_rcnn,
    lr=1e-5,
    start_epoch=6,
    epoches=11,
    save_weight_interval=1
    ):
    
    continue_training = True
    restart_ep = start_epoch
    restart_for_eps = epoches
    curr_model = faster_rcnn    
    
    while continue_training:
        result = train(
            curr_model,
            lr,
            restart_ep,
            restart_for_eps,
            save_weight_interval
        )
        if result is not None:
            message = result["message"]
            if message == "nan_loss":
                curr_epoch = result["curr_epoch"]
                if curr_epoch > (start_epoch + epoches):
                    print("stop training")
                    continue_training = False
                else:
                    continue_training = True
                    model_latest_epoch = (curr_epoch-1) - ((curr_epoch-1) % save_weight_interval)
                    restart_ep = model_latest_epoch + 1
                    restart_for_eps = epoches - (model_latest_epoch - start_epoch)
                    model_path = f"pths/model_epoch_{model_latest_epoch}.pth"
                    curr_model = FasterRCNNResnet50FPN().to(device)
                    curr_model.load_state_dict(torch.load(model_path))  
                    curr_model.train()  
                    
                    print(f"Get nan loss, restart training at epoch {restart_ep} for additional {restart_for_eps} epochs")
            else:
                continue_training = False
        else:
            continue_training = False


def main():
    model_path = "pths/model_epoch_41.pth"

    faster_rcnn = FasterRCNNResnet50FPN().to(device)
    
    if model_path is not None:
        faster_rcnn.load_state_dict(torch.load(model_path))
        
    faster_rcnn.train()
    
    train_with_nan(
        faster_rcnn,
        lr=1e-5,
        start_epoch=42,
        epoches=8,
        save_weight_interval=1
    )



if __name__ == "__main__":
    main()
