"""Makes a panoramic picture with two photos"""

# TIP: FEED PHOTOS IN LEFT TO RIGHT!!!!
# Remember to add file paths on lines 64 and 65!

import cv2
import numpy as np


def display(img, winname):
    # displays image in window named winname
    while True:

        cv2.imshow(f'{winname}', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def match(img1, img2):

    # Find key points
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match points
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    # Create a list of desirable points (distance may be modified to improve chances of a success)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    # Draws matches into an image
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # Preps photos to be combined
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w, _ = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # I don't know what this does yet
    #dst = cv2.perspectiveTransform(pts, M)

    # Combines the images
    dst = cv2.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2

    return dst, img3

# Read in the images
img1 = cv2.imread('')
img2 = cv2.imread('')


# Execute the code
try:
    pic, mat = match(img1, img2)
    
    # For resizing output
    # pic = cv2.resize(pic, (int(pic.shape[1] // 3), int(pic.shape[0] // 3)))
    # mat = cv2.resize(mat, (int(mat.shape[1] // 3), int(mat.shape[0] // 3)))

    display(pic, 1)

except:
    print("[-] There was an ERROR . . .")
    print('\n\n\n\n    . . . best guess is not enough matches were found in the images, try editing line 38 . . .')

