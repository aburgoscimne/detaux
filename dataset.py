from torch.utils.data import Dataset
import torchvision.transforms.functional as FT
import torch
import os
import cv2
import pandas as pd
from ast import literal_eval
from utils import *

class DetDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = FT.to_tensor(image)
        image_id = os.path.splitext(self.images[idx])[0]
        
        target = {}
        if self.split != 'test':
            boxes, labels = self.get_image_boxes_and_labels(image_id)
            target['boxes'] = boxes
            target['labels'] = labels

        return image, target, image_id
        
    def __len__(self):
        return len(self.images)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))

class WheatDataset(DetDataset):
    def __init__(self, root, split):
        super().__init__(root, split)
        self.images_dir = os.path.join(root, split)
        self.images = list(sorted(os.listdir(self.images_dir)))

        if split != 'test':
            self.boxes_df = pd.read_csv(os.path.join(root, split) + '.csv')

    def get_image_boxes_and_labels(self, image_id):
        boxes = self.boxes_df[self.boxes_df['image_id'] == image_id]['bbox'].apply(literal_eval)
        if len(boxes) > 0:
            boxes = torch.FloatTensor(list(boxes))
            boxes = x1y1wh_to_x1y1x2y2(boxes)
            labels = torch.ones([boxes.shape[0]], dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0), dtype=torch.int64)
        return boxes, labels