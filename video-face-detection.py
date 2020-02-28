import cv2

# File path may not work, I tried to fix it but need to check.
face_cascade = cv2.CascadeClassifier('Vision/haarcascades/haarcascade_frontalface_alt.xml')

#Detects a face using the haarcascade_frontalface_default template
def detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + w), (255, 255, 255), 5)

    return face_img


cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read(0)

    frame = detect_face(frame)

    cv2.imshow('stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
