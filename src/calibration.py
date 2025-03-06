import time
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from captura import *
from procesamiento import process_eye

# Configuraciones de ventana y variables para la calibración
calibration_points = [(60,60), (320,60), (580,60), (60,240), (320,240), (580,240), (60,420), (320,420), (580,420)]
calibration_data = []

# Modelos de regresión
model_left = LinearRegression()
model_right = LinearRegression()

def calibrate():
    start_time = time.time()
    while time.time() - start_time < 5:  # Mostrar mensaje por 3 segundos
        ret, frame = cap.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, 1)
        # Mostrar mensaje en pantalla
        cv2.putText(flipped_frame, "Mira los puntos de calibracion", (80, 250), font, font_scale, text_color, thickness)
        cv2.imshow("Frame", flipped_frame)
        cv2.waitKey(1)

    for point in calibration_points:
        print(f"Mira el punto en {point}")
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                left_pupil = process_eye(frame, landmarks, [37, 38, 39, 40, 41, 42])
                right_pupil = process_eye(frame, landmarks, [43, 44, 45, 46, 47, 48])
                if left_pupil and right_pupil:
                    calibration_data.append((left_pupil, right_pupil, point))
                    break
    train_models()

def train_models():
    # Convertir los datos de calibración de manera más eficiente
    X_left = np.array([[lp[0], lp[1]] for lp, _, _ in calibration_data])
    X_right = np.array([[rp[0], rp[1]] for _, rp, _ in calibration_data])
    y = np.array([point for _, _, point in calibration_data])
    model_left.fit(X_left, y)
    model_right.fit(X_right, y)
    print("Modelos entrenados.")