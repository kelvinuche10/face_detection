import cv2
from random import randrange


def face_detector(image_name):

    # load pre-trained data on face frontals from opencv (haarcascades frontal default)
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # choose an image to detect face with
    img = cv2.imread(image_name)

    # your webcam
    # webcam = cv2.VideoCapture(0)
    
    # convert to gray_scale
    gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # detect faces
    face_cordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    # for drawing rectangles around face
    for x, y, w, h in face_cordinates:
        cv2.rectangle(img, (x, y,), ( x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    # print(face_cordinates)

    # This shows the image in the argument
    cv2.imshow('face dectector', img)

    # this key delays the code until any key is entered
    cv2.waitKey()

    print('Code completed')

face_detector('second.jpg')