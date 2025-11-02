import cv2
import numpy as np




def harris_corners(reference_image):
    gray = cv2.cvtColor(reference_image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    reference_image[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite('assignment_4/Harris_corners.png', reference_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

img = cv2.imread("assignment_4/reference_img.png")
harris_corners(img)