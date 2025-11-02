import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



img1 = cv.imread('assignment_4/align_this.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('assignment_4/reference_img.png', cv.IMREAD_GRAYSCALE) # trainImage
MIN_MATCH_COUNT = 10
#MAX_FEATURES = 10 did not work assumed you meant min match count here
GOOD_MATCH_PERCENT = 0.7

def sift_align(image_to_align, reference_image, min_match_count=MIN_MATCH_COUNT, good_match_percent=GOOD_MATCH_PERCENT):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_to_align, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    
        h, w = image_to_align.shape
        h2, w2 = reference_image.shape
        

        img1_aligned = cv.warpPerspective(image_to_align, M, (w2, h2))

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
    
        reference_image = cv.polylines(reference_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        cv.imwrite('assignment_4/SIFT_aligned.png', img1_aligned) 
    
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                    singlePointColor=None,
                    matchesMask=matchesMask,  # draw only inliers
                    flags=2)
    
    img3 = cv.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)
    
    cv.imwrite('assignment_4/SIFT_output.png', img3)


sift_align(img1, img2, min_match_count=MIN_MATCH_COUNT, good_match_percent=GOOD_MATCH_PERCENT)