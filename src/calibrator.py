import cv2  
import numpy as np  
import time 
import csv  
import os  
from datetime import datetime  
from pupil_detector import PupilDetector  
import pyautogui  
from video_capture import VideoCapture 
import sys  

class Calibrator:
    def __init__(self, screen_width, screen_height):
        """Inicializa el calibrador. """
        # Guarda el ancho de la pantalla
        self.screen_width = screen_width  
        # Guarda el alto de la pantalla
        self.screen_height = screen_height  
        # Puntos de calibración
        self.calibration_points = [
            (180, 100), (960, 100), (1760, 100),
            (180, 540), (960, 540), (1760, 540),
            (180, 980), (960, 980), (1760, 980)
        ]
        # Duración del punto
        self.point_duration = 4 
        # Radio del punto
        self.radius = 15
        # Color del punto
        self.color = (0, 0, 255)
        # Borde del punto
        self.edge = 18
        # Relleno
        self.thickness = -1
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

    def show_instruction_message(self, cap, duration=3):
        """ Muestra un mensaje de instrucción inicial."""
        start_time = time.time()
        message = "Mire sin mover la cabeza a los puntos de la pantalla"
        
        while time.time() - start_time < duration:
            frame = cap.get_frame()
            if frame is None:
                continue
            
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
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            
            cv2.imshow('Calibracion', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def start_calibration(self, cap):
        """Inicia el proceso de calibración."""
        self.calibration_data = []
        self.current_point_index = 0
        self.calibrating = True
        self.point_started = False
        self.start_time = time.time()
        self.last_update_time = self.start_time
        print(f"Iniciando calibración con {len(self.calibration_points)} puntos")
        
        if not self.show_instruction_message(cap):
            print("Calibración interrumpida")
            self.calibrating = False

    def add_sample(self, left_pupil, right_pupil):
        """Añade una muestra de calibración."""
        if not self.calibrating:
            return False
        current_time = time.time()

        # Si es el inicio de un nuevo punto
        if not self.point_started and self.current_point_index < len(self.calibration_points):
            self.point_started = True
            self.last_update_time = current_time
            print(f"Comenzando punto {self.current_point_index + 1} de {len(self.calibration_points)}")
            self._add_calibration_point(left_pupil, right_pupil)
            return True

        # Si ya comenzó el punto actual
        if self.point_started and self.current_point_index < len(self.calibration_points):
            elapsed = current_time - self.last_update_time
            self._add_calibration_point(left_pupil, right_pupil)

            # Si ha pasado el tiempo para el punto actual, pasa al siguiente
            if elapsed >= self.point_duration:
                print(f"Punto {self.current_point_index + 1} completado")
                self.current_point_index += 1
                self.point_started = False

                # Si ya terminó todos los puntos, guarda los datos
                if self.current_point_index >= len(self.calibration_points):
                    self.calibrating = False
                    self.save_calibration_data()
                    return False
            return True
        return False

    def _add_calibration_point(self, left_pupil, right_pupil):
        """Añade un punto de calibración a los datos."""
        # Solo añade si ambos ojos tienen coordenadas válidas
        if left_pupil[0] is not None and right_pupil[0] is not None:
            target_x, target_y = self.calibration_points[self.current_point_index]  # Coordenadas objetivo
            sample = {
                'timestamp': time.time(),
                'target_x': target_x,
                'target_y': target_y,
                'left_pupil_x': left_pupil[0],
                'left_pupil_y': left_pupil[1],
                'right_pupil_x': right_pupil[0],
                'right_pupil_y': right_pupil[1]
            }
            self.calibration_data.append(sample)
        return True

    def save_calibration_data(self):
        """Guarda los datos de calibración en un archivo CSV."""
        if not self.calibration_data:
            return

        os.makedirs('../calibration', exist_ok=True)  # Crea el directorio si no existe
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Fecha y hora actual
        filename = f'../calibration/calibration_{timestamp}.csv'

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
        
        # Muestra mensaje de calibración completada
        self._show_completion_message(cap)
        
    def _show_completion_message(self, cap):
        """Muestra un mensaje de calibración completada durante 4 segundos."""

        start_time = time.time()
        message = "Calibracion completada"
        
        while time.time() - start_time < 4:  # Mostrar durante 4 segundos
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
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Dibujar el texto
            cv2.putText(frame, message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)
            
            # Mostrar el frame
            cv2.imshow('Calibracion', frame)
            cv2.waitKey(1)

    def draw_calibration_point(self, frame):
        """
        Dibuja el punto de calibración actual en el frame.
        Las coordenadas se escalan según el tamaño del frame actual.
        """
        if not self.calibrating or self.current_point_index >= len(self.calibration_points):
            return frame

        # Obtener dimensiones del frame actual
        frame_height, frame_width = frame.shape[:2]
        
        # Obtener el punto de calibración actual
        x, y = self.calibration_points[self.current_point_index]
        
        # Escalar las coordenadas al tamaño del frame actual
        x_scaled = int(x * frame_width / self.screen_width)
        y_scaled = int(y * frame_height / self.screen_height)
        
        # Tamaño del punto proporcional al tamaño del frame
        point_radius = max(20, int(min(frame_width, frame_height) * 0.01))
        
        # Copiar el frame para dibujar
        frame_copy = frame.copy()
        
        # Dibujar un círculo rojo más grande con borde blanco para mejor visibilidad
        cv2.circle(frame_copy, (x_scaled, y_scaled), point_radius + 3, (255, 255, 255), -1)  # Borde blanco
        cv2.circle(frame_copy, (x_scaled, y_scaled), point_radius, (0, 0, 255), -1)  # Relleno rojo
        
        # Mostrar progreso
        progress = f"Calibrando: Punto {self.current_point_index + 1}/{len(self.calibration_points)}"
        font_scale = max(0.5, min(frame_width, frame_height) / 1000) * 1.5
        font_thickness = max(1, int(font_scale * 1.5))
        
        # Calcular posición del texto para que sea visible
        text_size = cv2.getTextSize(progress, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = 10
        text_y = text_size[1] + 10
        
        # Fondo semitransparente para el texto
        cv2.rectangle(frame_copy, (0, 0), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        cv2.addWeighted(frame_copy, 0.7, frame_copy, 0.3, 0, frame_copy)
        
        cv2.putText(frame_copy, progress, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Mostrar cuenta regresiva si el punto ha comenzado
        if self.point_started:
            elapsed = time.time() - self.last_update_time
            countdown = max(0, int(self.point_duration - elapsed))
            
            # Tamaño del texto de la cuenta regresiva
            countdown_text = str(countdown + 1)
            
            # Calcular el tamaño de fuente basado en el radio del punto
            base_font_scale = point_radius / 20.0  
            countdown_scale = max(0.8, min(base_font_scale, 3.0)) 
            countdown_thickness = max(2, int(countdown_scale * 1.8))
            
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
                (255, 255, 255),
                countdown_thickness,
                cv2.LINE_AA
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
        calibrador.start_calibration(cap)
        
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
                        cv2.circle(frame, (left_pupil_x, left_pupil_y), 7, (0, 0, 255), -1)
                    
                    if right_pupil_x is not None and right_pupil_y is not None:
                        right_pupil = (right_pupil_x, right_pupil_y)
                        # Dibujar círculo en la pupila derecha
                        cv2.circle(frame, (right_pupil_x, right_pupil_y), 7, (0, 0, 255), -1)
                    
                # Añadir muestra de calibración
                calibrador.add_sample(left_pupil, right_pupil)
                
                # Dibujar el punto de calibración
                frame = calibrador.draw_calibration_point(frame)
                
                # Mostrar el frame
                cv2.imshow('Calibracion', frame)
                
                # Salir con 'esc'
                if cv2.waitKey(1) == 27:
                    break

        except KeyboardInterrupt:
            print("Calibración interrumpida manualmente.")

        finally:
            if calibrador.calibrating:
                calibrador.save_calibration_data()
            cap.release()
            cv2.destroyAllWindows()