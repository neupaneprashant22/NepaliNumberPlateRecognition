import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
video_path = '/Users/pneupane/Desktop/Screenshots/object_test.mov'

# Set the confidence threshold and NMS threshold for object detection
conf_threshold = 0.5
nms_threshold = 0.5

# Set the minimum size of the object bounding box
min_box_size = 50

# Open the video file
cap = cv2.VideoCapture(video_path)

# Loop through the frames in the video
while True:
    # Read the current frame
    ret, frame = cap.read()

    # If there are no more frames, exit the loop
    if not ret:
        break

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the YOLOv5 model to detect objects in the current frame
    results = model(frame)

    # Check if any objects were detected
    if len(results.xyxy) < 2:
        continue

    # Filter the predicted bounding boxes
    boxes = results.xyxy[0]
    scores = results.xyxy[1]
    class_ids = results.xyxy[2]

    # Apply confidence thresholding
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)

    # Filter out the overlapping boxes
    filtered_boxes = [boxes[idx[0]] for idx in indices]

    # Loop through the filtered boxes
    for box in filtered_boxes:
        # Get the box coordinates
        x1, y1, x2, y2 = map(int, box)

        # If the box size is less than the minimum size, skip this detection
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the video frame and the bounding boxes
    cv2.imshow('frame', frame)

    # Wait for a key press to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
