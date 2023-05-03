import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Load input image
img = cv2.imread("image.jpg")

# Perform inference on the input image
results = model(img)

# Extract bounding boxes of detected objects
boxes = results.xyxy[0].numpy()

# Save the bounding box of the first detected object to a new file
x1, y1, x2, y2, conf, class_id = boxes[0]
cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
cv2.imwrite("cropped_image.jpg", cropped_img)
