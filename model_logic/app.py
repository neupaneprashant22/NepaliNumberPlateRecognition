from fastapi import FastAPI,UploadFile,File
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
import torch
import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
import imutils
import sys
import os
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
from PIL import Image
import pathlib
import glob
from skimage import io, color, filters, morphology, measure
import nepali_roman as nr
from fastapi.middleware.cors import CORSMiddleware
import torchvision
import re

app = FastAPI()

# Define CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if(len(results_list)<4):
        return '0'
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

def getFrames(video_path):

    directory = 'Frames'
    if not os.path.exists(directory):
        os.makedirs(directory)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    interval = 3
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if count == int(fps * interval):
            filename = os.path.join(directory, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(filename, frame)
            count = 0
    cap.release()

def postProcess(recognized):
    english_converted=""
    prefix=""
    middle=""
    first_numbers=""
    last_numbers=""
    processed_plate=""
    first_numbers_count=0
    last_numbers_count=0
    try:
        if(nr.is_devanagari(recognized)):
            if(nr.romanize_text(recognized)):
                english_converted=nr.romanize_text(recognized)
            else:
                return '0'
        else:
            processed_plate=english_converted
    except Exception:
        return '0'  
    english_converted=english_converted.replace('०','0')
    if(english_converted[:2]!='ga' and english_converted[:2]!='ko' and english_converted[:2]!='na'):
        prefix='ba'
    else:
        prefix=english_converted[:2]
    if("pa" in english_converted):
        middle="pa"
    elif("kha" in english_converted):
        middle="kha"
    else:
        middle="cha"
    for x in english_converted:
        if x.isnumeric():
            first_numbers+=x
            first_numbers_count+=1
        if(first_numbers_count==2):
            break
    for x in reversed(english_converted):
        if x.isnumeric():
            last_numbers+=x
            last_numbers_count+=1
        if(last_numbers_count==4):
            break
    last_numbers= ''.join(reversed(last_numbers))
    processed_plate=prefix+first_numbers+middle+last_numbers
    if first_numbers_count==0:
        return '0'
    return processed_plate

def getCombined(easy,cnn):
    prefix=""
    middle=""
    first_numbers=""
    last_numbers=""
    easy_prefix=easy[:2]
    if("ga" in easy):
        easy_prefix="ga"
    if("pa" in easy):
        easy_middle="pa"
    elif("kha" in easy):
        easy_middle="kha"
    else:
        easy_middle="cha"
    if(len(easy)<4):
        return '0'
    if(len(cnn)<4):
        return '0'
    if(len(re.findall(r'\d+(?=[A-Za-z])', easy))>0):
        easy_first_numbers = re.findall(r'\d+(?=[A-Za-z])', easy)[0]
    if(len(easy_first_numbers))>2:
        easy_first_numbers=easy_first_numbers[:2]
    if(len(re.findall(r'[A-Za-z](\d+)', easy))>0):
        easy_last_numbers = re.findall(r'[A-Za-z](\d+)', easy)[-1]
    if(len(easy_last_numbers))>4:
        easy_last_numbers=easy_last_numbers[-4:]
    
    easy_modified=easy_prefix+easy_first_numbers+easy_middle+easy_last_numbers

    cnn_prefix=cnn[:2]
    cnn_middle="pa"
    test1=re.findall(r'[A-Za-z](\d+)', cnn)
    cnn_first_numbers=""
    cnn_last_numbers=""
    if(len(test1)>0):
      cnn_first_numbers = re.findall(r'[A-Za-z](\d+)', cnn)[0]
    if(len(cnn_first_numbers))>2:
        cnn_first_numbers=cnn_first_numbers[:2]
    if(len(re.findall(r'[A-Za-z](\d+)', cnn))>0):
        cnn_last_numbers = re.findall(r'[A-Za-z](\d+)', cnn)[-1]
    if(len(cnn_last_numbers))>4:
        cnn_last_numbers=cnn_last_numbers[-4:]
    cnn_modified=cnn_prefix+cnn_first_numbers+cnn_middle+cnn_last_numbers
    prefix="ba"
    first_numbers=cnn_first_numbers
    last_numbers=cnn_last_numbers
    middle=easy_middle
    if("ga" in easy):
        return easy
    if (cnn=='0'):
        return easy
    if ("cha" in easy):
        middle="cha"
    if(len(cnn)<9) and cnn[:-4].isnumeric():
        intermidiate_text=easy[:len(easy) - 4]+cnn[:-4]
        return intermidiate_text
    if(len(first_numbers)<1):
        first_numbers=easy_first_numbers
    if(len(last_numbers)<3):
        last_numbers=easy_last_numbers
    print(cnn_first_numbers)
    return (prefix+first_numbers+middle+last_numbers)


def getImageSegmantation(image):
    gray = color.rgb2gray(image)
    threshold = filters.threshold_otsu(gray)
    binary = gray > threshold
    opened = morphology.opening(binary, morphology.square(1))
    label_image = measure.label(opened, background=0)
    regions = measure.regionprops(label_image)
    text_regions = []
    for region in regions:
        if region.area > 50 and region.extent > 0.2 and region.extent < 0.9:
            text_regions.append(region)

    if not os.path.exists("output"):
        os.makedirs("output")

    for i, region in enumerate(text_regions):
        region_image = image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        region_image=cv2.resize(region_image,(150,150))
        io.imsave("output/region_{0}.jpg".format(i), region_image)


def imagePreprocessing(img):
    image = img
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5,5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed_image
def getProcessed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def deleteFiles():
    folder_path='./cropped_images/'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
def deleteFrames():
    folder_path='./Frames/'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
def deleteOutputImages():
    folder_path='./output/'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
def getBinary(image):
    img_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh,img_black=cv2.threshold(img_greyscale,100,255,cv2.THRESH_BINARY)
    cv2.imwrite('greyscale.jpg',img_black)
    IMAGE_PATH = './greyscale.jpg'
    img = cv2.imread(IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)
    return img

def ocrImage(img):
    values=[]
    reader = easyocr.Reader(['ne'])
    result = reader.readtext(img)
    for detection in result: 
        text = detection[1]
        values.append(text)
    return values

# Configure Cloudinary with your account credentials
cloudinary.config(
  cloud_name = "prashantneupane",
  api_key = "191114295379274",
  api_secret = "WKVuwJk4T0hdBT3VhaZx33khaWk"
)

@app.get("/get-frames")
async def separateVideo(video_url):
    getFrames(video_url)
    return {"message":"saved"}
@app.get("/image-detection")
async def detect_image(public_id):
    public_id=str(public_id)
    image_info = cloudinary.api.resource(public_id)

    image_url = image_info['url']

    response = requests.get(image_url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

    img = cv2.imread("image.jpg")

    results = model(img)

    boxes = results.xyxy[0].numpy()
    conf_scores = results.xyxy[0][:, 4].numpy()

    high_conf_boxes = boxes[conf_scores >= 0.1]
    print(high_conf_boxes)
    i=0
    for box in high_conf_boxes:
      x1, y1, x2, y2, conf, class_id = box
      cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
      cv2.imwrite("./cropped_images/cropped_image"+str(i)+".jpg", cropped_img)
      i=i+1
    results=[]
    
    folder_path = "./cropped_images"  
    converted_result=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            getImageSegmantation(img) 
            images_path=glob.glob('./output/*.jpg')
            pred_dict={}
            for i in images_path:
                region_image=cv2.imread(i)
                pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)
            print(pred_dict )
            ocr_result=ocrImage(img)
            result = "".join(ocr_result)
            after_result=postProcess(result)
            main_ocr=get_mainocr(img)
            print("After result: "+after_result)
            print("Main ocr "+main_ocr)
            combined_result=getCombined(after_result,main_ocr)
            converted_result.append(combined_result)
            converted_result.append(" ")
            deleteOutputImages()
    deleteFiles()
    return {"message": converted_result}
@app.post("/uploadvideo/")
async def upload_video(video: UploadFile = File(...)):
    video_bytes = await video.read()
    with open(video.filename, "wb") as f:
        f.write(video_bytes)
    return {video.filename}

@app.get("/video-detection")
def detect_video(video_url):
    getFrames("./"+video_url)
    video_results=[]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
 
    frame_path = "./Frames"  
    converted_result=[]
    for filename in os.listdir(frame_path):
        img = cv2.imread("./Frames/"+filename)

        results = model(img)

        boxes = results.xyxy[0].numpy()
        conf_scores = results.xyxy[0][:, 4].numpy()

        high_conf_boxes = boxes[conf_scores >= 0.1]
        print(high_conf_boxes)
    
        i=0
        for box in high_conf_boxes:
            x1, y1, x2, y2, conf, class_id = box
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite("./cropped_images/cropped_image"+str(i)+".jpg", cropped_img)
            i=i+1
        results=[]
        
        folder_path = "./cropped_images" 
        converted_result=[]
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                image_path = os.path.join(folder_path, filename)
                img = cv2.imread(image_path)
                getImageSegmantation(img) 
                images_path=glob.glob('./output/*.jpg')
                pred_dict={}
                for i in images_path:
                    region_image=cv2.imread(i)
                    pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)
                print(pred_dict )
                ocr_result=ocrImage(img)
                result = "".join(ocr_result)
                after_result=postProcess(result)
                main_ocr=get_mainocr(img)
                print("After result: "+after_result)
                print("Main ocr "+main_ocr)
                combined_result=getCombined(after_result,main_ocr)
                converted_result.append(combined_result)
                video_results.append(combined_result)
                video_results.append(" ")
                deleteOutputImages()
        deleteFiles()
    deleteFrames()
    return {"message": video_results}