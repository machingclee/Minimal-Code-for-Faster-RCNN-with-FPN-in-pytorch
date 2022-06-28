
import csv
import os
import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from src import config
from pydash import get, set_
from typing import List
from torchvision import transforms
from torchvision.transforms import ToPILImage
from copy import deepcopy

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transform = transforms.Compose([transforms.ToTensor()])


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (0, 0, w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


class AnnotationDataset(Dataset):
    def __init__(self, mode="train"):
        assert mode in ["train", "test"]
        self.mode = mode
        super(AnnotationDataset, self).__init__()
        self.annotations = {}
        self.cls_names = set()
        self.classnames = ["BG", "RBC", "Platelets", "WBC"]
        with open("dataset_blood/test.csv") as f:
            next(f)  # skip first line
            for line in f:
                line = line.split(",")
                img_basename, cls_name, x1, y1, x2, y2 = line
                xmin = float(x1)
                xmax = float(y1)
                ymin = float(x2)
                ymax = float(y2)
                cls_index = self.classnames.index(cls_name)

                box = [xmin, ymin, xmax, ymax, cls_index]

                if self.annotations.get(img_basename, None) is None:
                    self.annotations.update({img_basename: [box]})
                else:
                    self.annotations[img_basename].append(box)

            # [imagesbasename, [[bbox, cls_index]]]
            self.data: List[str, List[List[float]]] = [[k, np.array(v)] for k, v in self.annotations.items()]
            random.shuffle(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img_basename, boxes = data

        img_path = os.path.join(config.training_img_dir, img_basename)
        img = Image.open(img_path)
        img_pil_original = deepcopy(img)
        img = resize_and_padding(img)

        img = torch_img_transform(img)

        boxes_ = torch.as_tensor(boxes[..., 0:4]).float()
        targets = torch.as_tensor(boxes[..., 4]).float()

        # draw = ImageDraw.Draw(img_pil_original)
        # for box in boxes_:
        #     draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='Green')
        # img_pil_original.save("performance_check/test.jpg")

        if self.mode == "train":
            return img, boxes_, targets
        else:
            return img_pil_original, boxes_, targets

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    a, b = dataset[0]
    print(a, b)
