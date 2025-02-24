import cv2
import dlib

class EyeTracker:
    def __init__(self):
        # Inicializa el detector de caras y el predictor de puntos faciales
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('C:\\Users\\Alex\\Desktop\\shape_predictor_68_face_landmarks.dat')  # Necesitas descargar este archivo
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_landmarks(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta caras
        faces = self.face_detector(gray)
        for face in faces:
            # Encuentra los puntos faciales (68 puntos)
            landmarks = self.landmark_predictor(gray, face)
            return landmarks
        return None

    def process_frame(self, frame):
        landmarks = self.detect_landmarks(frame)
        if landmarks:
            for i in range(36, 42):  # Ojos: puntos 36-41
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)

        # Mostrar la imagen procesada
        cv2.imshow('Eye Tracker', frame)

import numpy as np

import numpy as np

class EyeGazeEstimator:
    def __init__(self):
        # Puntos de referencia en la cara (por ejemplo, los ojos)
        self.eye_model = np.array([  # Asumimos un modelo 3D simple para los ojos
            [0.0, 0.0, 0.0],  # punto en el ojo izquierdo
            [0.0, 0.0, 0.1],  # punto en el ojo derecho
        ])
        self.camera_matrix = np.array([
            [640, 0, 320],  # Suponiendo resolución de 640x480 y el centro en el punto medio
            [0, 480, 240],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((4, 1))
