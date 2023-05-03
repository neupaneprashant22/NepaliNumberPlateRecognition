import cv2

# Load the license plate image and the template
img = cv2.imread('cropped_image1.jpg', 0)
template = cv2.imread('~/Desktop/Screenshots/test-val2.png', 0)

# Apply template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Define the threshold for template matching
threshold = 0.8

# Find the locations of the matched template
locations = cv2.findNonZero(result > threshold)

# Draw rectangles around the matched templates
for loc in locations:
    x, y = loc[0]
    w, h = template.shape[::-1]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the result
cv2.imshow('Result', img)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', img)
