import cv2
import numpy as np
import dlib

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # Estados: (x, y, dx, dy) - Medidas: (x, y)
        
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def predict(self, coordX, coordY):
        """Usa la medición real si está disponible, sino predice."""
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        
        if not self.initialized:
            self.kf.statePre[:2] = measured
            self.kf.statePre[2:] = 0  # Velocidad inicial en 0
            self.initialized = True
        
        self.kf.correct(measured)  # Ajusta con la medición real
        predicted = self.kf.predict()  # Predice la próxima posición
        
        return int(predicted[0][0]), int(predicted[1][0])  # Extraer valores correctos

def get_gradient_edges(gray):
    """Detecta bordes en la imagen del ojo para encontrar la pupila."""
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    _, edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    return edges

def fit_ellipse(edges):
    """Encuentra la mejor elipse que se ajuste a los bordes detectados."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if len(cnt) >= 5]  # Necesario para fitEllipse
    
    if not contours:
        return None
    
    best_ellipse = None
    best_score = float('inf')
    
    for cnt in contours:
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma > MA else np.sqrt(1 - (ma / MA) ** 2)
        
        if 0.1 < eccentricity < 0.9:  # Filtrar elipses demasiado alargadas
            score = abs(ma - MA)  # Preferir formas más redondas
            if score < best_score:
                best_score = score
                best_ellipse = ellipse
    
    return best_ellipse

def process_eye(frame, landmarks, eye_indices, kalman):
    """Detecta la pupila y la filtra con Kalman."""
    x1, y1 = landmarks.part(eye_indices[0]).x, landmarks.part(eye_indices[0]).y
    x2, y2 = landmarks.part(eye_indices[3]).x, landmarks.part(eye_indices[3]).y
    eye_crop = frame[y1:y2, x1:x2]
    
    if eye_crop.size == 0:
        return None

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    edges = get_gradient_edges(gray)
    ellipse_params = fit_ellipse(edges)
    
    if ellipse_params is None:
        return None
    
    (cx, cy), _, _ = ellipse_params
    pupil_x, pupil_y = int(cx) + x1, int(cy) + y1

    # Filtrar con Kalman solo si la pupila es detectada
    return kalman.predict(pupil_x, pupil_y)

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\anton\\Downloads\\UNIVERSIDAD\\TFG\\shape_predictor_68_face_landmarks.dat")

kalman_left = KalmanFilter()
kalman_right = KalmanFilter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Ojo izquierdo: puntos 36 a 41
        left_pupil = process_eye(frame, landmarks, [37, 38, 39, 40, 41, 42], kalman_left)
        
        # Ojo derecho: puntos 42 a 47
        right_pupil = process_eye(frame, landmarks, [43, 44, 45, 46, 47, 48], kalman_right)

        if left_pupil:
            cv2.circle(frame, left_pupil, 3, (0, 255, 0), -1)
        if right_pupil:
            cv2.circle(frame, right_pupil, 3, (0, 255, 0), -1)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()