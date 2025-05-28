import cv2
import numpy as np

class VideoCapture:
    def __init__(self, camera_index=0, width=1920, height=1080):
        """
        Inicializa la captura de video con la configuración especificada.
        
        Args:
            camera_index (int): Índice de la cámara a utilizar.
            width (int): Ancho del frame de video.
            height (int): Alto del frame de video.
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
            
        self.width = width
        self.height = height
        
        # Configura resolución de la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self):
        """
        Lee un frame de la cámara.
        
        Returns:
            tuple: (ret, frame) donde ret es un booleano que indica si la lectura fue exitosa,
                 y frame es la imagen capturada.
        """
        return self.cap.read()
    
    def release(self):
        """Libera los recursos de la cámara."""
        self.cap.release()
    
    def is_opened(self):
        """
        Verifica si la cámara está abierta.
        
        Returns:
            bool: True si la cámara está abierta, False en caso contrario.
        """
        return self.cap.isOpened()
    
    def get_frame(self):
        """
        Obtiene un frame de la cámara.
        
        Returns:
            numpy.ndarray: El frame capturado o None si hay un error.
        """
        ret, frame = self.read()
        return frame if ret else None
    
    def __enter__(self):
        """Permite usar la clase con el contexto 'with'."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Asegura que los recursos se liberen correctamente."""
        self.release()
    
    def __iter__(self):
        """Permite iterar sobre los frames del video."""
        return self
    
    def __next__(self):
        """Obtiene el siguiente frame del video."""
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame