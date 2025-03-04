import cv2
import dlib

# Inicialización de la cámara y modelo de detección facial
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\anton\\Downloads\\UNIVERSIDAD\\TFG\\shape_predictor_68_face_landmarks.dat")