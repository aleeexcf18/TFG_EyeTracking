import time
import cv2
import numpy as np
from captura import *
from procesamiento import process_eye

# Configuraciones de ventana y variables para la calibración
calibration_points = [(60,60), (320,60), (580,60), (60,240), (320,240), (580,240), (60,420), (320,420), (580,420)]
calibration_data = []

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
        print(f"Mira el punto en ({point[0]},{point[1]})")
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            flipped_frame = cv2.flip(frame, 1)
            # Dibujar el punto de calibración en la pantalla
            cv2.circle(flipped_frame,point,10,(0,0,255),-1)

            cv2.imshow("Frame", flipped_frame)
            cv2.waitKey(1)

            for face in faces:
                landmarks = predictor(gray, face)
                left_pupil = process_eye(frame, landmarks, [37, 38, 39, 40, 41, 42])
                right_pupil = process_eye(frame, landmarks, [43, 44, 45, 46, 47, 48])
                if left_pupil and right_pupil:
                    calibration_data.append((left_pupil, right_pupil, point))
                    print(f"Pupilas registradas en: {left_pupil}, {right_pupil} para el punto {point}")
                    break

    # Mostrar mensaje de calibración finalizada
    print("Calibración finalizada")
    start_time = time.time()
    while time.time() - start_time < 3:  # Espera 3 segundos
        ret, frame = cap.read()
        if not ret:
            return
        flipped_frame = cv2.flip(frame, 1)
        cv2.putText(flipped_frame, "Calibracion finalizada", (150, 240), font, font_scale, (0, 255, 0), 2)
        cv2.imshow("Frame", flipped_frame)
        cv2.waitKey(1)
calibrate()