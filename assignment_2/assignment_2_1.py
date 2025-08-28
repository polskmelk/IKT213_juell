import cv2
from matplotlib import pyplot as plt
import numpy as np

def print_image_information(image_path):
    image = cv2.imread(image_path)

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    return height, width, channels, emptyPictureArray




def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("assignment_2/solutions/lena-1_padded.png", reflect)

def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("assignment_2/solutions/lena-1_cropped.png", cropped_image)


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("assignment_2/solutions/lena-1_resized.png", resized_image)



def copy(image, emptyPictureArray):
    emptyPictureArray[:] = image
    cv2.imwrite("assignment_2/solutions/lena-1_copied.png", emptyPictureArray)

def grayscale(image):
    cv2.imwrite("assignment_2/solutions/lena-1_grayscale.png", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def hsv(image):
    cv2.imwrite("assignment_2/solutions/lena-1_hsv.png", cv2.cvtColor(image, cv2.COLOR_BGR2HSV))


def hue_shifted(image, emptyPictureArray, hue):
    emptyPictureArray[:] = np.clip(image.astype(np.int16) + hue, 0, 255).astype(np.uint8)
    cv2.imwrite("assignment_2/solutions/lena-1_hue_shifted.png", emptyPictureArray)
    return emptyPictureArray


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    cv2.imwrite("assignment_2/solutions/lena-1_smoothed.png", smoothed_image)

def rotation(image, rotation_angle):
    rotated_image = cv2.rotate(image, rotation_angle)
    #cv2.imwrite("assignment_2/solutions/lena-1_rotated_90.png", rotated_image)
    cv2.imwrite("assignment_2/solutions/lena-1_rotated_180.png", rotated_image)

height, width, channels, emptyPictureArray = print_image_information("assignment_2/lena-1.png")
#padding(cv2.imread("assignment_2/lena-1.png"), 100)
#cropped_image = crop(cv2.imread("assignment_2/lena-1.png"), 80, width-130, 80, height-130)
#resize(cv2.imread("assignment_2/lena-1.png"), 200, 200)
#copy(cv2.imread("assignment_2/lena-1.png"), emptyPictureArray)
#grayscale(cv2.imread("assignment_2/lena-1.png"))
#hsv(cv2.imread("assignment_2/lena-1.png"))
#hue_shifted(cv2.imread("assignment_2/lena-1.png"), emptyPictureArray, 50)
#smoothing(cv2.imread("assignment_2/lena-1.png"))
#rotation(cv2.imread("assignment_2/lena-1.png"), cv2.ROTATE_90_CLOCKWISE)
rotation(cv2.imread("assignment_2/lena-1.png"), cv2.ROTATE_180)