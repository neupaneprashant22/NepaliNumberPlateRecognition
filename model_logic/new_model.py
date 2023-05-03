import cv2
import numpy as np

# Load the YOLOv4 configuration and weights
cfg_path = "./Own_cfg_and_weights/ANNPR.cfg"
weights_path = "./Own_cfg_and_weights/ANNPR.weights"
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the image to detect objects in
img = cv2.imread("image.jpg")
img = cv2.resize(img, (416, 416))
img = img / 255.0
# Create a blob from the input image and run it through the network
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# Postprocess the detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in indices:
    i = i[0]
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
