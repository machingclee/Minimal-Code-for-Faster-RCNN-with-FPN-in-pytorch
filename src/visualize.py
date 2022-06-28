import torch
import torch.nn as nn
from src.faster_rcnn_renset_fpn import FasterRCNNResnet50FPN
from src.dataset import AnnotationDataset, torch_img_transform, resize_and_padding
from PIL import ImageDraw
from copy import deepcopy
from src.device import device

def visualize(fast_rcnn: nn.Module = None, image_name=None):
    dataset = AnnotationDataset(mode="test")
    
    img, boxes, _ = next(iter(dataset))    
    
    with torch.no_grad():
        if fast_rcnn is None:
            fast_rcnn = FasterRCNNResnet50FPN().to(device)
            
        fast_rcnn.eval()
               
        img, padding_window, original_wh = resize_and_padding(img,return_window=True)
        img_ori = deepcopy(img)
        img = torch_img_transform(img).to(device)
        
        scores, boxes, cls_idxes, rois = fast_rcnn(img[None, ...])
        
        draw = ImageDraw.Draw(img_ori, 'RGBA')
        
        for roi in rois:
            draw.rectangle(((roi[0], roi[1]), (roi[2], roi[3])), outline=(255,255,255,150), width=1)
            
        for score, box, cls_idx in zip(scores, boxes, cls_idxes):
            # if score < 0.25:
            #     continue 
            xmin, ymin, xmax, ymax = box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue', width=1)
            draw.text(
                (xmin + 4, ymin + 1),
                "{}: {:.2f}".format(dataset.classnames[cls_idx.item()], score.item()), (255, 255, 255)
            )

        if image_name is None:
            imgname = "test.jpg"
        else:
            imgname = image_name 
            
        img_ori = img_ori.crop(padding_window)
        img_ori.resize(original_wh)
        img_ori.save("performance_check/{}".format(imgname))
        img_ori.save("performance_check/latest.jpg".format(imgname))
    
        fast_rcnn.train()
    
if __name__=="__main__":
    visualize()
        
        
        
    
    