import torch
import torchvision
import numpy as np
import cv2


def get_mainocr(img):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ocr_yolo.pt')

    classes = model.module.names if hasattr(model, 'module') else model.names
    results = model(img)

    # Sort the detected objects by y-axis of the bounding box
    results_sorted = sorted(results.xyxy[0], key=lambda x: x[1])
    results_list = []

    # Loop over the sorted detected objects and add them to the results list as dictionaries
    for i in range(len(results_sorted)):
        label = results.names[int(results_sorted[i][5])]
        score = results_sorted[i][4]
        x1, y1, x2, y2 = results_sorted[i][:4].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        result_dict = {'class': label, 'score': score, 'center_x': cx,'center_y': cy}
        results_list.append(result_dict)

    first_list=[]
    second_list=[]

    base_dict=results_list[0]
    for x in results_list:
        if(abs((base_dict['center_y'])-(x['center_y']))<=10):
            first_list.append(x)
        else:
            second_list.append(x)

    first_list=sorted(first_list, key=lambda x: x["center_x"])
    second_list=sorted(second_list, key=lambda x: x["center_x"])
    result=""
    for x in first_list:
        result+=str((x['class']))
    for x in second_list:
        result+=str((x['class']))
    return result

image_path="./cropped_images/cropped_image0.jpg"
img = cv2.imread(image_path)
print(get_mainocr(img))