import cv2
import numpy as np 
import dlib

class PupilDetector:
    def __init__(self, predictor_path="../utils/shape_predictor_68_face_landmarks.dat"):
        """ Inicializa el detector de pupilas con el predictor de puntos faciales. """
        self.detector = dlib.get_frontal_face_detector()  # Crea el detector de caras frontal de dlib
        self.predictor = dlib.shape_predictor(predictor_path)  # Carga el modelo de landmarks faciales
        
        # Índices de los puntos de referencia para los ojos
        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    
    def detect_face(self, frame):
        """ Detecta caras en el frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame a escala de grises
        return self.detector(gray, 0)  
    
    def get_eye_region(self, frame, landmarks, eye_points):
        """ Obtiene la región del ojo a partir de los landmarks faciales."""
        eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) 
                             for point in eye_points], np.int32) 
        return eye_region 
    
    def refinar_centro_pupila(self, gray_eye, centro_inicial, tam_region=20):

        x, y = centro_inicial

        # Limitar coordenadas al tamaño de la imagen
        x1 = max(x - tam_region, 0)
        y1 = max(y - tam_region, 0)
        x2 = min(x + tam_region, gray_eye.shape[1])
        y2 = min(y + tam_region, gray_eye.shape[0])

        roi = gray_eye[y1:y2, x1:x2]

        # Umbral local (simple)
        _, thresh = cv2.threshold(roi, np.mean(roi) - 5, 255, cv2.THRESH_BINARY_INV)

        # Buscar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Tomar el contorno más grande
            contorno_principal = max(contours, key=cv2.contourArea)

            if len(contorno_principal) >= 5:
                # Ajustar una elipse
                elipse = cv2.fitEllipse(contorno_principal)
                (cx, cy), (MA, ma), angle = elipse

                # Convertir a coordenadas globales
                centro_refinado = (int(cx + x1), int(cy + y1))
                return centro_refinado, elipse
        # Si no hay contornos válidos, devolver el inicial
        return centro_inicial, None
    
    def detect_pupil(self, eye_region, frame, eye_side):
        """Detecta la pupila en la región del ojo."""
        # Obtener el rectángulo delimitador del ojo
        x, y, w, h = cv2.boundingRect(eye_region)
        
        # Extraer la región del ojo
        eye = frame[y:y+h, x:x+w]
        
        if eye.size == 0:
            return None, None
            
        # Convertir a escala de grises
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        # Derivadas de primer orden
        Lx = cv2.Sobel(gray_eye, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(gray_eye, cv2.CV_64F, 0, 1, ksize=3)

        # Derivadas de segundo orden
        Lxx = cv2.Sobel(Lx, cv2.CV_64F, 1, 0, ksize=3)
        Lyy = cv2.Sobel(Ly, cv2.CV_64F, 0, 1, ksize=3)
        Lxy = cv2.Sobel(Lx, cv2.CV_64F, 0, 1, ksize=3)

        # Escalar común: (Lx^2 + Ly^2)
        mag2 = Lx**2 + Ly**2

        # Aproximar curvatura de isofotas
        denom = (Ly**2) * Lxx - 2 * Lx * Ly * Lxy + (Lx**2) * Lyy + 1e-8

        Dx = -(Lx * mag2) / (denom)
        Dy = -(Ly * mag2) / (denom)

        magnitud = Dx**2 + Dy**2

        # Encontrar el mínimo de curvatura
        _, _, min_loc, _ = cv2.minMaxLoc(magnitud)

        
        centro_inicial = min_loc

        # Refinar centro con contornos y elipse
        centro_refinado, elipse = self.refinar_centro_pupila(gray_eye, centro_inicial)

        return centro_refinado, centro_inicial, elipse
    