import cv2

"""Uses RETR_CCOMP and TRESH_BINARY_INV to detect contours."""

cap = cv2.VideoCapture(0)


def cont_marker(img):
    img_blur = cv2.medianBlur(img, 35)

    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_gray, 55, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:

            cv2.drawContours(img, contours, i, (255, 0, 255), 3)
    return img


while True:

    _, frame = cap.read()

    frame = cont_marker(frame)

    cv2.imshow('img_contours', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
