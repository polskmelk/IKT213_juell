import cv2
import numpy as np
 
# Read the original image
img = cv2.imread("assignment_3/lambo.png")
image = cv2.imread("assignment_3/shapes-1.png")
template = cv2.imread("assignment_3/shapes_template.jpg", 0)

def sobel_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    
    cv2.imwrite("assignment_3/solutions/lambo_sobelxy.png", sobelxy)

def canny_edge_detection(image, threshold1, threshold2):
    #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    canny = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    
    cv2.imwrite("assignment_3/solutions/lambo_canny.png", canny)


def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv2.imwrite("assignment_3/solutions/shapes_matched.png", image)
    #ser at den grønne firkanten ikke blir markert, mistenker at den er for nærme kanten av bildet slik at whitespace i template ikke matcher pga mangel av hvite pixler til venstre for firkanten. da blir vell det riktig?


def resize(image, scale_factor:int, up_or_down:str):
   rows, cols, _channels = map(int, image.shape)
   if up_or_down == "up":
       image = cv2.pyrUp(image, dstsize=(scale_factor*cols, scale_factor*rows))
   elif up_or_down == "down":
       image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))

   cv2.imwrite(f"assignment_3/solutions/lambo_resized_{up_or_down}.png", image)
   print("test")

#sobel_edge_detection(img)
#canny_edge_detection(img, 100, 200)
#template_match(image, template)
resize(img, 2, "up")
resize(img, 2, "down")
