import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import random

class DetAux():
    """Auxiliary base."""
    def __init__(self, classes):
        self.classes = classes
        self.classes_map = {k: v + 1 for v, k in enumerate(self.classes)}
        self.classes_map['background'] = 0
        self.reverse_classes_map = {v: k for k, v in self.classes_map.items()}
        self.classes_colors = tuple(map(tuple, np.random.random((len(self.classes_map),3)) * 256))
        self.classes_color_map = {k: self.classes_colors[i] for i, k in enumerate(self.classes_map.keys())}

    def draw_bounding_boxes(self, image, boxes, labels):
        """
        Draw bounding boxes over an image.

        Parameters
        image : 3D (H,W,C) numpy array
        boxes : 2D list
            All boxes for an image in the following format:
                - x top-left
                - y top-left
                - width
                - height
        labels : 2D list
        """

        # OpenCV works with x1y1wh format
        boxes = x1y1x2y2_to_x1y1wh(boxes)

        for i,b in enumerate(boxes):
            x = int(b[0])
            y = int(b[1])
            w = int(b[2])
            h = int(b[3])
            box_color = self.classes_color_map[self.reverse_classes_map[labels[i]]]
            cv2.rectangle(image, (x,y), (x+w,y+h), box_color, 2)

            # The following commented code shows labels over the boxes
            # text_size = cv2.getTextSize(self.reverse_classes_map[labels[i]], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # cv2.rectangle(image, (x,y), (x+text_size[0]+1,y+text_size[1]+3), box_color, -1)
            # cv2.putText(image, self.reverse_classes_map[labels[i]], (x,y+text_size[1]+2), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    def show_image(self, image, size=(8,8), show_legend=True, image_id=None):
        """Show a loaded image"""
        plt.figure(figsize=size)
        plt.imshow(image)

        if image_id is not None:
            plt.title(image_id)

        if show_legend:
            patches = []
            for c in self.classes:
                color = self.classes_color_map[c]
                color = [x/255 for x in color] # Patch requires normalized RGB
                patches.append(mpatches.Patch(color=color, label=c))
            plt.legend(handles=patches)

    def show_batch_of_images(self, data_loader, limit=4):
        images, targets, idxs = next(iter(data_loader))

        for i in range(0,min(len(images), limit)):
            image = reverse_to_tensor(images[i])
            if len(targets[i]['boxes']) > 0:
                boxes = targets[i]['boxes'].numpy()
                labels = targets[i]['labels'].numpy()
                self.draw_bounding_boxes(image, boxes, labels)
            self.show_image(image, image_id=idxs[i])

def set_seed(seed):
    """Set random seed for experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def reverse_to_tensor(image):
    """Inverse action than to_tensor"""
    image = np.ascontiguousarray(image.permute(1,2,0).numpy())
    image = (image*255).astype(np.uint8)
    return image

def x1y1x2y2_to_x1y1wh(boxes):
    """
    Converts boxes from x1y1x2y2 (upper left, bottom right) format 
    to x1y1wh (upper left, width and height) format
    """
    boxes[:,2] = boxes[:,2] - boxes[:,0]
    boxes[:,3] = boxes[:,3] - boxes[:,1]
    return boxes

def x1y1wh_to_x1y1x2y2(boxes):
    """
    Converts boxes from x1y1wh (upper left, width and height) format 
    to x1y1x2y2 (upper left, bottom right) format
    """
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    x2 = x1 + w
    y2 = y1 + h
    boxes = torch.stack([x1,y1,x2,y2], dim=1)
    return boxes


def intersection_over_union(x1_p, y1_p, x2_p, y2_p, x1_g, y1_g, x2_g, y2_g):
    """Computes IoU between two boxes (predicted box and ground truth box)"""
    # Determine the (x, y)-coordinates of the intersection rectangle
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    # Compute the area of intersection rectangle
    i_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    
    # Compute the area of both the prediction and ground-truth rectangles
    p_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    g_area = (x2_g - x1_g + 1) * (y2_g - y1_g + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = i_area / float(p_area + g_area - i_area)

    return iou