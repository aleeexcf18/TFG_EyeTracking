import cv2
import dlib
import numpy as np

def landmarks():
    # Inicializar el detector de caras de dlib
    detector = dlib.get_frontal_face_detector()
    
    # Cargar el predictor de landmarks faciales
    predictor = dlib.shape_predictor("../utils/shape_predictor_68_face_landmarks.dat")
    
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    
    print("Presiona 'q' para salir")
    
    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break
        
        # Convertir a escala de grises para la detección de caras
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras en la imagen en escala de grises
        faces = detector(gray)
        
        # Dibujar los landmarks para cada cara detectada
        for face in faces:
            # Obtener los landmarks faciales
            landmarks = predictor(gray, face)
            
            # Dibujar los puntos de referencia
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                
                # Dibujar un círculo en cada punto de referencia
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Mostrar el número del punto
                cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Mostrar el frame en pantalla completa
        cv2.namedWindow('Deteccion de Rostro y Landmarks', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Deteccion de Rostro y Landmarks', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Deteccion de Rostro y Landmarks', frame)
        
        # Salir con 'esc'
        if cv2.waitKey(1) == 27:
            break
    
    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    landmarks()