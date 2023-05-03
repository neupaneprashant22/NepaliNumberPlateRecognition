from skimage import io, color, filters, morphology, measure
import os


def getImageSegmantation(image):
# Load image and convert to grayscale
    gray = color.rgb2gray(image)

    # Apply thresholding to convert to binary image
    threshold = filters.threshold_otsu(gray)
    binary = gray > threshold

    # Apply morphological opening to remove small objects and smooth edges
    opened = morphology.opening(binary, morphology.square(1))

    # Label connected components
    label_image = measure.label(opened, background=0)

    # Filter connected components to keep only those likely to contain text
    regions = measure.regionprops(label_image)
    text_regions = []
    for region in regions:
        if region.area > 50 and region.extent > 0.2 and region.extent < 0.9:
            text_regions.append(region)

    # Extract bounding boxes for text regions and save as separate image files
    if not os.path.exists("output"):
        os.makedirs("output")

    for i, region in enumerate(text_regions):
        region_image = image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        io.imsave("output/region_{0}.jpg".format(i), region_image)

image = io.imread("cropped_images/cropped_image0.jpg")
getImageSegmantation(image)