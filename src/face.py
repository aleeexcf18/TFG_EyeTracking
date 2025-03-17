#Detector 1 ojo
import cv2
import numpy as np
import dlib
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\anton\\Downloads\\UNIVERSIDAD\\TFG\\shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks = predictor(gray,face)
        left_point = (landmarks.part(36).x,landmarks.part(36).y)
        right_point = (landmarks.part(39).x,landmarks.part(39).y)
        center_top = (landmarks.part(37).x,landmarks.part(38).y)
        center_bottom = (landmarks.part(41).x,landmarks.part(40).y)
        rect_left = left_point[0]
        rect_right = right_point[0]
        rect_top = min(left_point[1], landmarks.part(38).y)
        rect_bottom = max(right_point[1], landmarks.part(40).y)
        cv2.rectangle(frame,(rect_left,rect_top),(rect_right,rect_bottom),(0,255,0),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == 13:
        break
cap.release()
cv2.destroyAllWindows()