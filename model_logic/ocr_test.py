import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import sys


def imagePreprocessing(img):
    image = img
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5,5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed_image

def getBinary(image):
    img_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh,img_black=cv2.threshold(img_greyscale,64,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite('np2.jpg',img_black)
    IMAGE_PATH = './np2.jpg'
    img = cv2.imread(IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)
    return img

def getProcessed(image):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def ocrImage(img):
    values=[]
    reader = easyocr.Reader(['ne','en'])
    result = reader.readtext(img)
    for detection in result: 
        text = detection[1]
        values.append(text)
    return values





if __name__ == "__main__":
   
    IMAGE_PATH = sys.argv[1]
    print((IMAGE_PATH))
    # IMAGE_PATH='/Users/pneupane/Desktop/test-2.png'
    img = cv2.imread(IMAGE_PATH)
    final_image1=imagePreprocessing(img)
    final_image2=getBinary(img)
    final_image3=getProcessed(img)
    # print(ocrImage(final_image1))
    # print(ocrImage(final_image2))
    print(ocrImage(final_image3))

