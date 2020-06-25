import cv2
import matplotlib.pyplot as plt

def draw_bounding_boxes(image, boxes):
    '''Draw bounding boxes over an image'''
    for b in boxes:
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    return image

def load_image_from_path(image_path:str):
    '''Load an image from a path'''
    image = cv2.imread(image_path)
    return image

def show_image(image, size=(5,5)):
    '''Show a loaded image'''
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))