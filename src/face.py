import cv2
import dlib
from capture import *

# Crear ventanas fijas para cada ojo
cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL)

def extract_eye(eye_points, facial_landmarks, frame):
    # Extrae la imagen del ojo usando los puntos de referencia

    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    rect_left = left_point[0]
    rect_right = right_point[0]
    rect_top = min(left_point[1], facial_landmarks.part(eye_points[2]).y)
    rect_bottom = max(right_point[1], facial_landmarks.part(eye_points[4]).y)
    
    # Dibujar rectángulo en cada ojo
    cv2.rectangle(frame,(rect_left,rect_top),(rect_right,rect_bottom),(0,255,0),2)

    # Extraer la región del ojo
    if rect_top >= 0 and rect_bottom <= frame.shape[0] and rect_left >= 0 and rect_right <= frame.shape[1]:
        eye_crop = frame[rect_top:rect_bottom, rect_left:rect_right]
        return eye_crop
    return None

while True:
    ret,frame = cap.read()
    if not ret:
        break
    
    flipped_frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped_frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()
        cv2.rectangle(flipped_frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks = predictor(gray, face)

        # Obtener la imagen recortada del ojo
        left_eye_crop = extract_eye([36, 37, 38, 39, 40, 41], landmarks, flipped_frame)
        right_eye_crop = extract_eye([42, 43, 44, 45, 46, 47], landmarks, flipped_frame)

        # Mostrar cada ojo en su ventana si fue detectado correctamente
        if left_eye_crop is not None:
            cv2.imshow("Left Eye", left_eye_crop)
        if right_eye_crop is not None:
            cv2.imshow("Right Eye", right_eye_crop)
            
    cv2.imshow("Frame",flipped_frame)
    if cv2.waitKey(1) & 0xFF == 13:
        break
cap.release()
cv2.destroyAllWindows()