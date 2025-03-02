#Modelo predicción
import cv2
import numpy as np
import dlib
import time
from sklearn.linear_model import LinearRegression

# Inicialización de cámara y modelo de detección facial
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\anton\\Downloads\\UNIVERSIDAD\\TFG\\shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_color = (0, 0, 255)
thickness = 2

# Configuración de ventana
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Puntos de calibración y datos
calibration_points = [(60,60), (320,60), (580,60), (60,240), (320,240), (580,240), (60,420), (320,420), (580,420)]
calibration_data = []

# Modelo de regresión para mapear pupilas a coordenadas en pantalla
model_left = LinearRegression()
model_right = LinearRegression()

# Función para detectar la pupila
def detect_pupil(eye_crop):
    gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # h, w = eye_crop.shape[:2]
   # cx, cy = w // 2, h // 2  # Centro predeterminado
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0: # Representa el áre del contorno detectado
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    
    return cx, cy

# Función para procesar el ojo
def process_eye(frame, landmarks, eye_indices):
    x1, y1 = landmarks.part(eye_indices[0]).x, landmarks.part(eye_indices[1]).y
    x2, y2 = landmarks.part(eye_indices[3]).x, landmarks.part(eye_indices[5]).y
    eye_crop = frame[y1:y2, x1:x2]
    
    if eye_crop.size == 0:
        return None
    
    cx, cy = detect_pupil(eye_crop)
    return (cx + x1, cy + y1)

# Fase de calibración
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
            flipped_frame = cv2.flip(frame, 1)
            
            cv2.circle(flipped_frame,point,10,(0,0,255),-1) # Punto de calibración en la pantalla
            
            for face in faces:
                landmarks = predictor(gray, face)
                left_pupil = process_eye(frame, landmarks, [36, 37, 38, 39, 40, 41])
                right_pupil = process_eye(frame, landmarks, [42, 43, 44, 45, 46, 47])
    
                if left_pupil and right_pupil:
                    calibration_data.append((left_pupil, right_pupil, point))
                    print(f"Pupilas registradas en: {left_pupil}, {right_pupil} para el punto {point}")
                    break
            
            cv2.imshow("Frame", flipped_frame)
            if cv2.waitKey(1) & 0xFF == 13: # presiona ENTER
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
    train_models()

# Entrenamiento del modelo
def train_models():
    X_left = np.array([[lp[0], lp[1]] for lp, _, _ in calibration_data])
    X_right = np.array([[rp[0], rp[1]] for _, rp, _ in calibration_data])
    y = np.array([point for _, _, point in calibration_data])
    
    model_left.fit(X_left, y)
    model_right.fit(X_right, y)
    print("Modelos entrenados.")

# Modo de predicción
def predict_gaze(left_pupil, right_pupil):
    left_pred = model_left.predict([[left_pupil[0], left_pupil[1]]])[0]
    right_pred = model_right.predict([[right_pupil[0], right_pupil[1]]])[0]
    return (int((left_pred[0] + right_pred[0]) / 2), int((left_pred[1] + right_pred[1]) / 2))
    
# Bucle principal
def main_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        flipped_frame = cv2.flip(frame, 1)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_pupil = process_eye(frame, landmarks, [36, 37, 38, 39, 40, 41])
            right_pupil = process_eye(frame, landmarks, [42, 43, 44, 45, 46, 47])
            
            if left_pupil and right_pupil:
                gaze_x, gaze_y = predict_gaze(left_pupil, right_pupil)
                cv2.circle(flipped_frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.putText(flipped_frame, f"Mirada: ({gaze_x}, {gaze_y})", (50, 50), font, font_scale, text_color, thickness)
       
        cv2.imshow("Frame", flipped_frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

calibrate()
main_loop()

cap.release()
cv2.destroyAllWindows()