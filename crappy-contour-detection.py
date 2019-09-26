import cv2
import imutils

cap = cv2.VideoCapture(0)


def get_lines(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 15)
    img_edged = cv2.Canny(img_blur, 35, 200)
    return img_edged


def draw_cont(edged_img, img):
    contours = cv2.findContours(edged_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    drawn_img = cv2.drawContours(img, contours[:], -1, (255, 0, 255), 2)
    return drawn_img


while True:

    _, frame = cap.read()

    edged_img = get_lines(frame)
    frame = draw_cont(edged_img, frame)

    cv2.imshow('stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

