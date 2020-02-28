import cv2
import numpy as np

"""Watershed algorithm on video"""

# Kernel to be used in image processing.
kernel = np.ones((3, 3), np.uint8)

# Functions


def gray_blur(img, blur_k):
    # Changes images color to gray and blurs based on kernel size selected.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.medianBlur(img_gray, blur_k)
    return img_gray_blur


def noise_removal(img):
    # Removes white noise from image using morphology.
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 2)
    return opening


def get_background(img):
    # Returns image with clear background.
    return cv2.dilate(img, kernel, 3)


def get_foreground(img):
    # Uses distance transformation and thresholding to create a certain area.
    img = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(img, img.max() * 0.5, 255, 0)
    return fg


def get_markers(img, unknown):
    # Finds seeds and makes unknown visable.
    _, markers = cv2.connectedComponents(img)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers


def draw_contours(base_img, modified_img):
    # Draws external contours on image.
    contours, hierarchy = cv2.findContours(modified_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(frame, contours, i, (255, 0, 255), 3)
    return frame


# Code

def watershed(img):

# Can be used to resize images.
# img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5))

    img_gb = gray_blur(img, 5)

    _, thresh = cv2.threshold(img_gb, 165, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    nr_img = noise_removal(thresh)

    certain_bg = get_background(nr_img)

    certain_fg = get_foreground(nr_img)

    certain_fg = np.uint8(certain_fg)
    unknown = cv2.subtract(certain_bg, certain_fg)

    markers = get_markers(certain_fg, unknown)

    markers = cv2.watershed(img, markers)

    final_img = draw_contours(img, markers)

    return final_img


cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    finished_frame = watershed(frame)

    cv2.imshow('stream', finished_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
