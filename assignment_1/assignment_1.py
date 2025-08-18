import cv2 as cv

def print_image_information(image_path):
    image = cv.imread(image_path)

  
    height, width, channels = image.shape
    print(f"Image Height: {height}")
    print(f"Image Width: {width}")
    print(f"Number of Channels: {channels}")
    print(f"Image Size: {image.size} bytes")
    print(f"Image Data Type: {image.dtype}")

print_image_information("assignment_1/lena-1.png")