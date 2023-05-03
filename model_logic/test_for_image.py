from PIL import Image, ImageOps
from scipy.ndimage import median_filter
import numpy as np

# Open image file
test_image_path="./cropped_images/cropped_image0.jpg"
img = Image.open(test_image_path)

# Convert to grayscale
img_gray = img.convert('L')

# Invert colors
img_invert = ImageOps.invert(img_gray)

# Convert black to white and white to black
# img_bw = img_invert.convert('1')

img_processed = median_filter(img_invert, size=3)
img_processed = Image.fromarray(img_processed, mode='L')
# Save image
# img_bw.save("new_image_file.png")
img_processed.save("processed.png")
