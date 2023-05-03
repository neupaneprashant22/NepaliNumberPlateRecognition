import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2
from skimage import io, color, filters, morphology, measure
import os

pred_path='./cropped_images'


classes=['0','1','2','3','4','5','6','7','8','9','ba','pa']

#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=output.view(-1,32*75*75)
        output=self.fc(output)
            
        return output


checkpoint=torch.load('best_checkpoint.model')
model=ConvNet(num_classes=12)
model.load_state_dict(checkpoint)
model.eval()

#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

#prediction function
def prediction(img_path,transformer):
    image=Image.open(img_path)   
    image_tensor=transformer(image).float()  
    image_tensor=image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()  
    input=Variable(image_tensor)
    output=model(input)
    index=output.data.numpy().argmax()
    pred=classes[index]
    return pred


def getImageSegmantation(image):
    gray = color.rgb2gray(image)

    # Apply thresholding to convert to binary image
    threshold = filters.threshold_otsu(gray)
    binary = gray > threshold

    # Apply morphological opening to remove small objects and smooth edges
    opened = morphology.opening(binary, morphology.square(3))

    # Label connected components
    label_image = measure.label(opened, background=0)

    # Filter connected components to keep only those likely to contain text
    regions = measure.regionprops(label_image)
    text_regions = []
    for region in regions:
        if region.area > 100 and region.extent > 0.2 and region.extent < 0.9:
            text_regions.append(region)

    # Extract bounding boxes for text regions and save as separate image files
    if not os.path.exists("output"):
        os.makedirs("output")

    # Create a copy of the original image for visualization
    image_with_rectangles = image.copy()

    # Draw rectangles on the image and save each text region
    for i, region in enumerate(text_regions):
        # Check if the region overlaps with any of the previous regions
        overlaps = False
        for j in range(i):
            if region.coords.tolist() in text_regions[j].coords.tolist():
                overlaps = True
                break
        
        # Draw rectangle on the image and save the text region
        if not overlaps:
            region_image = image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
            io.imsave("output/region_{0}.jpg".format(i), region_image)
            rect = measure.regionprops((region_image > filters.threshold_otsu(region_image)))[0].bbox
            rect = (rect[0] + region.bbox[0], rect[1] + region.bbox[1], rect[2] + region.bbox[0], rect[3] + region.bbox[1])
            image_with_rectangles[rect[0]:rect[2], rect[1]] = (1, 0, 0)
            image_with_rectangles[rect[0]:rect[2], rect[3]] = (1, 0, 0)
            image_with_rectangles[rect[0], rect[1]:rect[3]] = (1, 0, 0)
            image_with_rectangles[rect[2], rect[1]:rect[3]] = (1, 0, 0)

    # Save the image with rectangles
    io.imsave("image_with_rectangles.png", image_with_rectangles)



directory = "cropped_images"

# Loop through all files in the directory
for filename in os.listdir(directory):
    img=cv2.imread(filename)
    print("filename"+filename)
    getImageSegmantation(img)
    images_path=glob.glob('./output/*.jpg')
    print(images_path)

    pred_dict={}
    for i in images_path:
        print(prediction(i,transformer))
        pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)

    print(pred_dict)

    folder_path='./output/'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(file_path)
            os.remove(file_path)