import cv2
import numpy as np
 
# Read the original image
img = cv2.imread("assignment_3/lambo.png") 

def sobel_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    
    cv2.imwrite("assignment_3/solutions/lambo_sobelxy.png", sobelxy)
    print("Sobel Edge Detection applied and saved as 'lambo_sobelxy.png'")

def canny_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    
    cv2.imwrite("assignment_3/solutions/lambo_canny.png", canny)
    print("Canny Edge Detection applied and saved as 'lambo_canny.png'")




sobel_edge_detection(img)