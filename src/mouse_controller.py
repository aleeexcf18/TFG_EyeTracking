import cv2
import dlib
import numpy as np
import pyautogui
import time
from scipy.spatial import distance

class MouseController:
    def __init__(self):
        """Inicializa el controlador del mouse con detección facial usando dlib."""
        # Inicializar el detector de caras de dlib
        self.detector = dlib.get_frontal_face_detector()
        
        # Cargar el predictor de landmarks faciales
        self.predictor = dlib.shape_predictor("../utils/shape_predictor_68_face_landmarks.dat")
        
        # Índices de los puntos de referencia para los ojos
        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]  # Puntos del ojo izquierdo
        self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]  # Puntos del ojo derecho
        
        # Obtener dimensiones de la pantalla
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Configuración de PyAutoGUI
        pyautogui.FAILSAFE = False  # Desactivar la protección de seguridad
        
        # Configuración de la cámara
        self.cap = None
        
        # Umbral para detección de parpadeo
        self.EYE_AR_THRESH = 0.20
        self.BLINK_CONSEC_FRAMES = 2
        self.COUNTER = 0
    
    def start(self):
        """Inicia el control del mouse con la cámara."""
        try:
            # Inicializar la cámara
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")
                
            print("Control de mouse activado. Presiona 'esc' para salir.")
            
            while True:
                # Leer frame de la cámara
                success, frame = self.cap.read()
                if not success:
                    print("Error al capturar el frame")
                    break
                    
                # Procesar el frame
                self._process_frame(frame)
                
                # Mostrar el frame
                cv2.imshow('Control del Raton', frame)
                
                # Salir con 'esc'
                if cv2.waitKey(1) == 27:
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def _process_frame(self, frame):
        """Procesa un frame para detectar el movimiento ocular y controlar el mouse."""
        
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = self.detector(gray)
        
        for face in faces:
            # Obtener los landmarks faciales
            landmarks = self.predictor(gray, face)
            
            # Procesar los ojos y controlar el mouse
            self._process_eyes(landmarks, frame)
    
    def _process_eyes(self, landmarks, frame):
        """Procesa ambos ojos para controlar el puntero del mouse."""
        # Obtener las regiones de los ojos
        left_eye = self._get_eye_region(landmarks, self.LEFT_EYE_POINTS)
        right_eye = self._get_eye_region(landmarks, self.RIGHT_EYE_POINTS)
        
        # Calcular el centro de los ojos
        left_center = self._get_eye_center(left_eye)
        right_center = self._get_eye_center(right_eye)
        
        # Calcular la posición media entre los dos ojos
        if left_center and right_center:
            center_x = int((left_center[0] + right_center[0]) / 2)
            center_y = int((left_center[1] + right_center[1]) / 2)
            
            # Mapear las coordenadas del ojo a la pantalla
            screen_x = np.interp(center_x, [0, frame.shape[1]], [0, self.screen_w])
            screen_y = np.interp(center_y, [0, frame.shape[0]], [0, self.screen_h])
            
            # Mover el puntero del mouse
            pyautogui.moveTo(screen_x, screen_y)
        
        # Detectar parpadeo para hacer clic
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_POINTS)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_POINTS)
        
        # Promedio de la relación de aspecto de ambos ojos
        ear = (left_ear + right_ear) / 2.0
        
        # Detectar parpadeo
        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1
            if self.COUNTER >= self.BLINK_CONSEC_FRAMES:
                pyautogui.click()
                time.sleep(0.5)  # Pequeña pausa para evitar múltiples clics
        else:
            self.COUNTER = 0
    
    def _get_eye_region(self, landmarks, eye_points):
        """Obtiene la región del ojo a partir de los landmarks faciales."""
        eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) 
                             for point in eye_points], np.int32)
        return eye_region
    
    def _get_eye_center(self, eye_points):
        """Calcula el centro del ojo a partir de los puntos de referencia."""
        if len(eye_points) == 0:
            return None
        x = np.mean(eye_points[:, 0])
        y = np.mean(eye_points[:, 1])
        return (int(x), int(y))
    
    def _calculate_ear(self, landmarks, eye_points):
        """Calcula la relación de aspecto del ojo (EAR - Eye Aspect Ratio)."""
        try:
            # Obtener las coordenadas de los puntos del ojo
            points = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points])
            
            # Calcular las distancias verticales
            vert1 = distance.euclidean(points[1], points[5])
            vert2 = distance.euclidean(points[2], points[4])
            
            # Calcular la distancia horizontal
            horiz = distance.euclidean(points[0], points[3])
            
            # Evitar división por cero
            if horiz == 0:
                return 0.0
                
            # Calcular EAR
            ear = (vert1 + vert2) / (2.0 * horiz)
            return ear
            
        except Exception as e:
            print(f"Error al calcular EAR: {e}")
            return 0.0


def main():
    """Función principal para ejecutar el controlador del mouse."""
    controller = MouseController()
    controller.start()


if __name__ == "__main__":
    main()