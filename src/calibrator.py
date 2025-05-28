import cv2  # OpenCV para procesamiento de imágenes y visualización
import numpy as np  # Para operaciones numéricas
import time  # Para medir tiempos y duraciones
import csv  # Para guardar los datos de calibración
import os  # Para manejo de archivos y directorios
from datetime import datetime  # Para generar nombres de archivo con timestamp
from pupil_detector import PupilDetector  # Importar el detector de pupilas
import pyautogui  # Para obtener el tamaño de la pantalla
from video_capture import VideoCapture  # Nuestra clase personalizada para la cámara
import sys  # Para obtener el tamaño de la pantalla

class Calibrator:
    def __init__(self, screen_width, screen_height, calibration_points=9, point_duration=4):
        """
        Inicializa el calibrador.
        """
        # Guarda el ancho de la pantalla
        self.screen_width = screen_width  
        # Guarda el alto de la pantalla
        self.screen_height = screen_height  
        # Número de puntos de calibración
        self.calibration_points = calibration_points  
        # Duración de cada punto de calibración (segundos)
        self.point_duration = point_duration  
        # Genera los puntos de calibración
        self.points = self._generate_calibration_points()  
        # Lista para almacenar datos recogidos
        self.calibration_data = []  
        # Índice del punto de calibración actual
        self.current_point_index = 0  
        # Tiempo de inicio de la calibración
        self.start_time = 0  
        # Última vez que se actualizó el punto
        self.last_update_time = 0  
        # Estado de calibración
        self.calibrating = False  
        # Si el punto actual ya comenzó
        self.point_started = False  

    def _generate_calibration_points(self):
        """
        Genera las coordenadas de los puntos de calibración.
        Los puntos se distribuyen en una cuadrícula 3x3 relativa al tamaño de la pantalla.
        """
        # Calcular márgenes y espaciado basado en el tamaño de la pantalla
        margin_x = int(self.screen_width * 0.1)  # 10% de margen en los bordes
        margin_y = int(self.screen_height * 0.1)
        
        # Calcular el espacio entre puntos
        spacing_x = (self.screen_width - 2 * margin_x) // 2
        spacing_y = (self.screen_height - 2 * margin_y) // 2
        
        # Generar puntos en una cuadrícula 3x3
        points = []
        for i in range(3):
            for j in range(3):
                x = margin_x + j * spacing_x
                y = margin_y + i * spacing_y
                points.append((x, y))
        
        # Devolver solo los puntos necesarios según el parámetro
        return points[:self.calibration_points]

    def show_instruction_message(self, cap, duration=3):
        """
        Muestra un mensaje de instrucción centrado, mientras la cámara sigue mostrando al usuario en vivo.
        """
        start_time = time.time()
        message = "Mire sin mover la cabeza a los puntos de la pantalla"
        
        while time.time() - start_time < duration:
            frame = cap.get_frame()
            if frame is None:
                continue  # Espera hasta que haya un frame válido
            
            height, width = frame.shape[:2]
            font_scale = max(0.8, min(width, height) / 800) * 1.5
            thickness = max(1, int(font_scale * 1.5))
            
            (text_width, text_height), _ = cv2.getTextSize(
                message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2

            overlay = frame.copy()
            cv2.rectangle(overlay, 
                        (text_x - 20, text_y - text_height - 20),
                        (text_x + text_width + 20, text_y + 20),
                        (0, 0, 0), -1)
            
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            cv2.putText(frame, message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            
            cv2.imshow('Calibracion', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    
    def get_current_frame(self):
        """
        Obtiene el frame actual de la cámara.
        
        Returns:
            numpy.ndarray: El frame actual o None si hay un error.
        """
        # Esta función debe ser implementada o modificada según cómo se obtengan los frames
        # en tu aplicación. Por ahora, devuelve None como marcador de posición.
        return None
    
    def start_calibration(self):
        """Inicia el proceso de calibración."""
        self.calibration_data = []  # Limpiar datos anteriores
        self.current_point_index = 0
        self.calibrating = True
        self.point_started = False
        self.start_time = time.time()
        self.last_update_time = self.start_time  # Marca que el punto aún no ha comenzado
        self.start_time = time.time()  # Marca el tiempo de inicio
        self.last_update_time = self.start_time  # Marca la última actualización
        print(f"Iniciando calibración con {len(self.points)} puntos")
        
        # Mostrar mensaje de instrucción antes de comenzar
        instruction_shown = self.show_instruction_message(cap, duration=3)
        if not instruction_shown:
            print("Calibración interrumpida")
            return

    def add_sample(self, left_pupil, right_pupil):
        """
        Añade una muestra de calibración.
        """
        if not self.calibrating:
            return False  # Si no está calibrando, no hace nada

        current_time = time.time()  # Tiempo actual

        # Si es el inicio de un nuevo punto
        if not self.point_started and self.current_point_index < len(self.points):
            self.point_started = True  # Marca que el punto comenzó
            self.last_update_time = current_time  # Actualiza el tiempo
            print(f"Comenzando punto {self.current_point_index + 1} de {len(self.points)}")
            self._add_calibration_point(left_pupil, right_pupil)  # Añade la muestra actual
            return True

        # Si ya comenzó el punto actual
        if self.point_started and self.current_point_index < len(self.points):
            elapsed = current_time - self.last_update_time  # Tiempo transcurrido en el punto
            self._add_calibration_point(left_pupil, right_pupil)  # Añade la muestra actual

            # Si ha pasado el tiempo para el punto actual, pasa al siguiente
            if elapsed >= self.point_duration:
                print(f"Punto {self.current_point_index + 1} completado")
                self.current_point_index += 1  # Avanza al siguiente punto
                self.point_started = False  # Marca que el nuevo punto no ha comenzado

                # Si ya terminó todos los puntos, guarda los datos
                if self.current_point_index >= len(self.points):
                    print("Guardando datos de calibración...")
                    self.calibrating = False  # Termina la calibración
                    self.save_calibration_data()  # Guarda los datos
                    return False
            return True

        return False

    def _add_calibration_point(self, left_pupil, right_pupil):
        """Añade un punto de calibración a los datos."""
        # Solo añade si ambos ojos tienen coordenadas válidas
        if left_pupil[0] is not None and right_pupil[0] is not None:
            target_x, target_y = self.points[self.current_point_index]  # Coordenadas objetivo
            sample = {
                'timestamp': time.time(),
                'target_x': target_x,
                'target_y': target_y,
                'left_pupil_x': left_pupil[0],
                'left_pupil_y': left_pupil[1],
                'right_pupil_x': right_pupil[0],
                'right_pupil_y': right_pupil[1]
            }
            self.calibration_data.append(sample)  # Añade la muestra a la lista
        return True

    def save_calibration_data(self):
        """Guarda los datos de calibración en un archivo CSV."""
        if not self.calibration_data:
            return  # Si no hay datos, no hace nada

        os.makedirs('../calibration', exist_ok=True)  # Crea el directorio si no existe
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Fecha y hora actual
        filename = f'../calibration/calibration_{timestamp}.csv'  # Nombre del archivo

        # Escribe los datos en el archivo CSV
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'target_x', 'target_y',
                          'left_pupil_x', 'left_pupil_y',
                          'right_pupil_x', 'right_pupil_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self.calibration_data:
                writer.writerow(sample)
        print(f"Datos de calibración guardados en {filename}")
        
        # Mostrar mensaje de calibración completada
        self._show_completion_message(cap)
        
    def _show_completion_message(self, cap):
        """Muestra un mensaje de calibración completada durante 4 segundos.
        
        Args:
            cap: Instancia de VideoCapture para obtener los frames de la cámara.
        """
        start_time = time.time()
        message = "Calibracion completada"
        
        while time.time() - start_time < 4:  # Mostrar durante 4 segundos
            # Obtener un frame de la cámara
            frame = cap.get_frame()
            if frame is None:
                continue
                
            height, width = frame.shape[:2]
            font_scale = max(0.8, min(width, height) / 800) * 2.0  # Tamaño de fuente más grande
            thickness = max(2, int(font_scale * 1.5))
            
            # Calcular tamaño del texto
            (text_width, text_height), _ = cv2.getTextSize(
                message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Calcular posición centrada
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2

            # Crear un fondo semitransparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                        (text_x - 20, text_y - text_height - 20),
                        (text_x + text_width + 20, text_y + 20),
                        (0, 0, 0), -1)
            
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Dibujar el texto
            cv2.putText(frame, message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)  # Texto en verde
            
            # Mostrar el frame
            cv2.imshow('Calibracion', frame)
            cv2.waitKey(1)

    def draw_calibration_point(self, frame):
        """
        Dibuja el punto de calibración actual en el frame.
        Las coordenadas se escalan según el tamaño del frame actual.
        """
        if not self.calibrating or self.current_point_index >= len(self.points):
            return frame  # Si no está calibrando, retorna el frame sin cambios

        # Obtener dimensiones del frame actual
        frame_height, frame_width = frame.shape[:2]
        
        # Obtener el punto de calibración actual (en coordenadas de pantalla)
        x, y = self.points[self.current_point_index]
        
        # Calcular factores de escala
        scale_x = frame_width / self.screen_width
        scale_y = frame_height / self.screen_height
        
        # Escalar las coordenadas al tamaño del frame actual
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        
        # Tamaño del punto proporcional al tamaño del frame (aumentado un poco)
        point_radius = max(20, int(min(frame_width, frame_height) * 0.01))
        
        # Copiar el frame para dibujar
        frame_copy = frame.copy()
        
        # Dibujar un círculo rojo más grande con borde blanco para mejor visibilidad
        cv2.circle(frame_copy, (x_scaled, y_scaled), point_radius + 3, (255, 255, 255), -1)  # Borde blanco
        cv2.circle(frame_copy, (x_scaled, y_scaled), point_radius, (0, 0, 255), -1)  # Relleno rojo
        
        # Mostrar progreso
        progress = f"Calibrando: Punto {self.current_point_index + 1}/{len(self.points)}"
        font_scale = max(0.5, min(frame_width, frame_height) / 1000) * 1.5
        font_thickness = max(1, int(font_scale * 1.5))
        
        # Calcular posición del texto para que sea visible
        text_size = cv2.getTextSize(progress, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = 10
        text_y = text_size[1] + 10
        
        # Fondo semitransparente para el texto
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (0, 0), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
        
        cv2.putText(frame_copy, progress, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Mostrar cuenta regresiva si el punto ha comenzado
        if self.point_started and self.current_point_index < len(self.points):
            elapsed = time.time() - self.last_update_time
            countdown = max(0, int(self.point_duration - elapsed))
            
            # Tamaño del texto de la cuenta regresiva (reducido para que quepa en el punto)
            countdown_text = str(countdown + 1)
            
            # Calcular el tamaño de fuente basado en el radio del punto (aumentado)
            base_font_scale = point_radius / 20.0  # Reducido el divisor para aumentar el tamaño
            countdown_scale = max(0.8, min(base_font_scale, 3.0))  # Aumentado el tamaño máximo
            countdown_thickness = max(2, int(countdown_scale * 1.8))  # Aumentado el grosor
            
            # Calcular tamaño del texto para centrarlo
            (text_w, text_h), baseline = cv2.getTextSize(
                countdown_text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                countdown_scale, 
                countdown_thickness
            )
            
            # Calcular la posición para centrar el texto en el punto
            text_x = x_scaled - text_w // 2
            text_y = y_scaled + text_h // 2 - baseline // 2
            
            # Dibujar el número de cuenta regresiva centrado en el punto
            cv2.putText(
                frame_copy, 
                countdown_text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                countdown_scale, 
                (255, 255, 255),  # Color blanco
                countdown_thickness,
                cv2.LINE_AA  # Mejor calidad de texto
            )
        return frame_copy
    
if __name__ == "__main__":
    # Inicializar la cámara
    try:
        cap = VideoCapture(camera_index=0, width=1920, height=1080)
        print("Cámara iniciada correctamente")
    except RuntimeError as e:
        print(f"Error al iniciar la cámara: {e}")
    else:
        # Inicializar el detector de pupilas
        pupil_detector = PupilDetector("../utils/shape_predictor_68_face_landmarks.dat")
        
        # Obtener el tamaño de la pantalla
        screen_w, screen_h = pyautogui.size()
        
        # Inicializar el calibrador
        calibrador = Calibrator(screen_width=screen_w, screen_height=screen_h)
        
        # Configurar la ventana como pantalla completa
        cv2.namedWindow('Calibracion', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibracion', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Iniciar la calibración
        calibrador.start_calibration()
        
        # Bucle principal
        try:
            while True:
                # Leer un frame de la cámara
                frame = cap.get_frame()
                if frame is None:
                    print("Error: No se pudo capturar el frame.")
                    cv2.destroyAllWindows()
                    sys.exit(1)
                
                # Detectar caras
                faces = pupil_detector.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
                
                # Inicializar coordenadas de las pupilas
                left_pupil = (None, None)
                right_pupil = (None, None)
                
                if len(faces) > 0:
                    # Obtener landmarks faciales
                    landmarks = pupil_detector.predictor(frame, faces[0])
                    
                    # Obtener regiones de los ojos
                    left_eye_region = pupil_detector.get_eye_region(frame, landmarks, pupil_detector.LEFT_EYE_POINTS)
                    right_eye_region = pupil_detector.get_eye_region(frame, landmarks, pupil_detector.RIGHT_EYE_POINTS)
                    
                    # Detectar pupilas
                    left_pupil_x, left_pupil_y = pupil_detector.detect_pupil(left_eye_region, frame, 'left', landmarks)
                    right_pupil_x, right_pupil_y = pupil_detector.detect_pupil(right_eye_region, frame, 'right', landmarks)
                    
                    if left_pupil_x is not None and left_pupil_y is not None:
                        left_pupil = (left_pupil_x, left_pupil_y)
                        # Dibujar círculo en la pupila izquierda
                        cv2.circle(frame, (left_pupil_x, left_pupil_y), 5, (0, 0, 255), -1)
                    
                    if right_pupil_x is not None and right_pupil_y is not None:
                        right_pupil = (right_pupil_x, right_pupil_y)
                        # Dibujar círculo en la pupila derecha
                        cv2.circle(frame, (right_pupil_x, right_pupil_y), 5, (0, 0, 255), -1)
                
                # Añadir muestra de calibración
                calibrador.add_sample(left_pupil, right_pupil)
                
                # Dibujar el punto de calibración
                frame = calibrador.draw_calibration_point(frame)
                
                # Mostrar el frame
                cv2.imshow('Calibracion', frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Solo guardar si no se completó la calibración normalmente
            if calibrador.calibrating:
                calibrador.save_calibration_data()
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()