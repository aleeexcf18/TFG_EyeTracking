import cv2
import dlib
import numpy as np
import pyautogui

# Cargar detector de rostros y predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\anton\\Downloads\\UNIVERSIDAD\\TFG\\shape_predictor_68_face_landmarks.dat")

# Obtener dimensiones de pantalla
screen_width, screen_height = pyautogui.size()

# Función para obtener el centro de un ojo
def get_eye_center(eye_points, facial_landmarks):
    x = sum(facial_landmarks.part(point).x for point in eye_points) // len(eye_points)
    y = sum(facial_landmarks.part(point).y for point in eye_points) // len(eye_points)
    return (x, y)

# Puntos de referencia para los ojos (según shape_predictor_68)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_center = get_eye_center(LEFT_EYE, landmarks)
        right_eye_center = get_eye_center(RIGHT_EYE, landmarks)
        
        # Promediar ambos ojos para estimar dirección de la mirada
        gaze_x = (left_eye_center[0] + right_eye_center[0]) // 2
        gaze_y = (left_eye_center[1] + right_eye_center[1]) // 2
        
        # Mapear coordenadas de la cámara a la pantalla
        screen_x = np.interp(gaze_x, [0, frame.shape[1]], [0, screen_width])
        screen_y = np.interp(gaze_y, [0, frame.shape[0]], [0, screen_height])
        
        # Mover el cursor del ratón
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        
        # Dibujar puntos en los ojos
        cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)
    
    cv2.imshow("Eye Tracking Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()