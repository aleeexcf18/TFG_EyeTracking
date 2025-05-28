import cv2
import numpy as np 
import dlib  # Importa dlib para detección de caras y landmarks faciales

class PupilDetector:
    def __init__(self, predictor_path="../utils/shape_predictor_68_face_landmarks.dat"):
        """
        Inicializa el detector de pupilas con el predictor de puntos faciales.
        
        Args:
            predictor_path (str): Ruta al archivo del predictor de puntos faciales de dlib.
        """
        self.detector = dlib.get_frontal_face_detector()  # Crea el detector de caras frontal de dlib
        self.predictor = dlib.shape_predictor(predictor_path)  # Carga el modelo de landmarks faciales
        
        # Índices de los puntos de referencia para los ojos (según convención dlib 68 puntos)
        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    
    def detect_face(self, frame):
        """
        Detecta caras en el frame.
        
        Args:
            frame: Imagen en la que detectar caras.
            
        Returns:
            list: Lista de caras detectadas.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame a escala de grises
        return self.detector(gray, 0)  # Devuelve la lista de caras detectadas
    
    def get_eye_region(self, frame, landmarks, eye_points):
        """
        Obtiene la región del ojo a partir de los landmarks faciales.
        
        Args:
            frame: Imagen de entrada.
            landmarks: Puntos de referencia faciales.
            eye_points: Índices de los puntos del ojo.
            
        Returns:
            numpy.ndarray: Región del ojo.
        """
        eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) 
                             for point in eye_points], np.int32)  # Extrae las coordenadas de los puntos del ojo
        return eye_region  # Devuelve el polígono que delimita la región del ojo
    
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
        """
        Detecta la pupila en la región del ojo.
        
        Args:
            eye_region: Región del ojo.
            frame: Imagen completa.
            eye_side (str): 'left' o 'right' para identificar el ojo.
            
        Returns:
            tuple: (centro_x, centro_y) de la pupila, o (None, None) si no se detecta.
        """
        # Obtener el rectángulo delimitador del ojo
        x, y, w, h = cv2.boundingRect(eye_region)  # Calcula el bounding box de la región del ojo
        
        # Extraer la región del ojo
        eye = frame[y:y+h, x:x+w]  # Recorta la región del ojo del frame
        
        if eye.size == 0:
            return None, None  # Si la región está vacía, retorna None
            
        # Convertir a escala de grises
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # Convierte el ojo a escala de grises
        
        # Derivadas de primer orden
        Lx = cv2.Sobel(gray_eye, cv2.CV_64F, 1, 0, ksize=3)
        Ly = cv2.Sobel(gray_eye, cv2.CV_64F, 0, 1, ksize=3)

        # Derivadas de segundo orden
        Lxx = cv2.Sobel(Lx, cv2.CV_64F, 1, 0, ksize=3)
        Lyy = cv2.Sobel(Ly, cv2.CV_64F, 0, 1, ksize=3)
        Lxy = cv2.Sobel(Lx, cv2.CV_64F, 0, 1, ksize=3)  # igual a Ly respecto a x

        # Escalar común: (Lx^2 + Ly^2)
        mag2 = Lx**2 + Ly**2

        # Paso 4: Aproximar curvatura de isofotas (simplificación)
        denom = (Ly**2) * Lxx - 2 * Lx * Ly * Lxy + (Lx**2) * Lyy + 1e-8

        Dx = -(Lx * mag2) / (denom)
        Dy = -(Ly * mag2) / (denom)

        magnitud = Dx**2 + Dy**2

        # Paso 5: Encontrar el mínimo de curvatura (posible centro pupila)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(magnitud)

        '''
        centro_inicial = min_loc

        # Paso 6: Refinar centro con contornos y elipse
        centro_refinado, elipse = self.refinar_centro_pupila(gray_eye, centro_inicial)

        return centro_refinado, centro_inicial, elipse
        '''
        center_x = min_loc[0] + x
        center_y = min_loc[1] + y
            
        return center_x, center_y 
    