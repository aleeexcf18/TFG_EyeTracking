import cv2
import numpy as np

def detect_pupil(eye_region):
    # Convertir la región del ojo a escala de grises
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para detectar el círculo de la pupila
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Buscar círculos en la región de la pupila usando el detector HoughCircles
    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dibuja el círculo de la pupila detectada
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(eye_region, center, radius, (0, 255, 0), 2)  # Círculo verde
            cv2.circle(eye_region, center, 2, (0, 0, 255), 3)  # Centro rojo
