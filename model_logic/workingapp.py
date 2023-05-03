from fastapi import FastAPI
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

app = FastAPI()
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
    reader = easyocr.Reader(['ne','en'])
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

# Define a FastAPI endpoint that retrieves and saves the image
@app.get("/download-image")
async def download_image(public_id):
    public_id=str(public_id)
    image_info = cloudinary.api.resource(public_id)

    # Get the URL of the image
    image_url = image_info['url']

    # Download the image and save it to a file in the current directory
    response = requests.get(image_url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

    # Load input image
    img = cv2.imread("image.jpg")
    # img=getProcessed(img)

    # Perform inference on the input image
    results = model(img)

    # Extract bounding boxes and confidence scores of detected objects
    boxes = results.xyxy[0].numpy()
    conf_scores = results.xyxy[0][:, 4].numpy()

    # Filter out objects with a confidence score below 0.7
    high_conf_boxes = boxes[conf_scores >= 0.1]
    print(high_conf_boxes)
    # Save the bounding boxes of the high-confidence objects to a new file
    i=0
    for box in high_conf_boxes:
      x1, y1, x2, y2, conf, class_id = box
      cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
      cv2.imwrite("./cropped_images/cropped_image"+str(i)+".jpg", cropped_img)
      i=i+1
    results=[]
    
    folder_path = "./cropped_images"  # Replace with the path to your folder

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Replace extensions with the ones you want to read
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
        final_image=imagePreprocessing(img)
        final_image=getBinary(img)
        last_image=getProcessed(img)
        print(ocrImage(last_image))
        result_value=ocrImage(last_image)
        results.append(result_value)
    # deleteFiles()
    return {"message": "Number Plate detected as:"+str(results)}
