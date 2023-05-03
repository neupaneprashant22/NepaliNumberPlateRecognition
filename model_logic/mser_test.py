import cv2

# Read image
img = cv2.imread("processed.png")

# Convert to grayscale

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold image to create binary image
_, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to remove noise and connect text regions
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours in the image
contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size and aspect ratio
min_width = img.shape[1] * 0.05  # Minimum width of text region
max_width = img.shape[1] * 0.9   # Maximum width of text region
min_height = img.shape[0] * 0.03 # Minimum height of text region
max_height = img.shape[0] * 0.5  # Maximum height of text region
filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if aspect_ratio > 0.1 and aspect_ratio < 10 and w > min_width and w < max_width and h > min_height and h < max_height:
        filtered_contours.append(contour)

# Draw bounding boxes on image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Display image
cv2.imwrite("mser.jpg",img)