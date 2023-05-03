import torch
import torchvision
from PIL import Image
import cv2
import numpy as np

def getClassName(val):
    if(val=='1'):
        return '0'
    elif(val=='2'):
        return '1'
    elif(val=='3'):
        return '2'
    elif(val=='4'):
        return '3'
    elif(val=='5'):
        return '4'
    elif(val=='6'):
        return '5'
    elif(val=='7'):
        return '6'
    elif(val=='8'):
        return '7'
    elif(val=='9'):
        return '8'
    elif(val=='10'):
        return '9'
    elif(val=='11'):
        return 'ba'
    elif(val=='12'):
        return 'pa'

def detect_class_yolov5(image_path):
    # Load the YOLOv5 model
    model_classes=['0','1','2','3','4','5','6','7','8','9','ba','pa']
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ocr_yolo.pt')

    # Load the image and convert it to a tensor
    img = cv2.imread(image_path)

    # Run the image through the model to detect objects
    results = model(img)

    # Extract detected objects' labels and positions
    labels, positions = results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, :4].numpy()

    # Get the top-left corner of each bounding box
    top_lefts = positions[:, :2]

    # Get the rows and columns of the top-left corners
    rows, cols = top_lefts[:, 1], top_lefts[:, 0]

    # Sort the indices of the top-left corners in ascending order
    top_left_indices = np.lexsort((cols, rows))

    # Get the bottom-right corner of each bounding box
    bottom_rights = positions[:, 2:]

    # Get the rows and columns of the bottom-right corners
    rows, cols = bottom_rights[:, 1], bottom_rights[:, 0]

    # Sort the indices of the bottom-right corners in ascending order
    bottom_right_indices = np.lexsort((cols, rows))

    # Append labels from top-left to right
    output_str = ""
    for i in top_left_indices:
        # Append the label to the output string
        intermidiate=str(int(labels[i]))
        print(intermidiate)
        getClassName(intermidiate)
        output_str += getClassName(intermidiate) + " "

    # Append labels from bottom-left to right
    for i in bottom_right_indices:
        # Append the label to the output string if it hasn't been added already
        if str(labels[i]) not in output_str:
            intermidiate=str(int(labels[i]))
            getClassName(intermidiate)
            print(intermidiate)
            output_str += getClassName(intermidiate) + " "

        # Return the output string
    return output_str
   

image_path="./cropped_images/cropped_image0.jpg"
print(detect_class_yolov5(image_path))