import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
#MAX_FEATURES = 10
GOOD_MATCH_PERCENT = 0.7

img1 = cv.imread('assignment_4/align_this.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('assignment_4/reference_img.png', cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector with max features
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < GOOD_MATCH_PERCENT * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
 
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
 
    #h, w = img1.shape
    h2, w2 = img2.shape
    
    # Align img2 to match img1
    img1_aligned = cv.warpPerspective(img1, M, (w2, h2))

    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
 
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    cv.imwrite('assignment_4/SIFT_aligned.png', img1_aligned) 
 
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
 
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
 
cv.imwrite('assignment_4/SIFT_output.png', img3)
