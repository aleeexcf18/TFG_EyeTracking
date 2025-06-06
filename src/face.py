import cv2
import dlib
import numpy as np
from video_capture import VideoCapture

def face():
    # Inicializar el detector de caras de dlib
    detector = dlib.get_frontal_face_detector()
    
    # Cargar el predictor de landmarks faciales
    predictor = dlib.shape_predictor("../utils/shape_predictor_68_face_landmarks.dat")
    
    # Iniciar la cámara usando nuestra clase VideoCapture
    try:
        cap = VideoCapture()
        print("Cámara iniciada correctamente")
    except RuntimeError as e:
        print(f"Error al iniciar la cámara: {e}")
        return
    
    print("Presiona 'q' para salir")
    
    while True:
        # Leer un frame de la cámara
        frame = cap.get_frame()
        if frame is None:
            print("Error: No se pudo capturar el frame.")
            break
        
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras en la imagen
        faces = detector(gray)
        
        for face in faces:
            # Obtener las coordenadas del rectángulo de la cara
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Dibujar rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Obtener los landmarks faciales
            landmarks = predictor(gray, face)
            
            # Coordenadas de los ojos
            # Ojo izquierdo: puntos 36 a 41
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            # Ojo derecho: puntos 42 a 47
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
            # Función para dibujar rectángulo alrededor de los ojos
            def draw_eye_rect(eye_points, frame, color):
                x_min = min(p[0] for p in eye_points)
                x_max = max(p[0] for p in eye_points)
                y_min = min(p[1] for p in eye_points)
                y_max = max(p[1] for p in eye_points)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Dibujar rectángulos alrededor de los ojos 
            draw_eye_rect(left_eye_points, frame, (0, 255, 0))
            draw_eye_rect(right_eye_points, frame, (0, 255, 0))
        
        # Mostrar el frame en pantalla completa
        cv2.namedWindow('Deteccion del Rostro', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Deteccion del Rostro', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Deteccion del Rostro', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face()