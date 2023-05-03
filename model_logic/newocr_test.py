import cv2

# Load the image
img = cv2.imread('./cropped_images/cropped_image0.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert white pixels to black and every other color to white
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]


smoothed = cv2.GaussianBlur(thresh, (5, 5), 20)

# Save the result
cv2.imwrite('result.jpg', thresh)
cv2.imwrite('result1.jpg',smoothed)
