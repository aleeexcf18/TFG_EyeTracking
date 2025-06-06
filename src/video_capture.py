import cv2
import numpy as np

class VideoCapture:
    def __init__(self, camera_index=0, width=1920, height=1080):
        """Inicializa la captura de video con la configuración especificada."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
            
        self.width = width
        self.height = height
        
        # Configurar resolución de la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self):
        """Lee un frame de la cámara."""
        return self.cap.read()
    
    def release(self):
        """Libera los recursos de la cámara."""
        self.cap.release()
    
    def is_opened(self):
        """Verifica si la cámara está abierta."""
        return self.cap.isOpened()
    
    def get_frame(self):
        """Obtiene un frame de la cámara."""
        ret, frame = self.read()
        return frame if ret else None