import cv2
import numpy as np
import time
import dlib
from scipy.spatial import distance  # Para calcular distancias entre puntos
import sys

class PupilDetector:
    def __init__(self, predictor_path="../utils/shape_predictor_68_face_landmarks.dat"):
        """Inicializa el detector de pupilas con el predictor de puntos faciales."""
        self.detector = dlib.get_frontal_face_detector()  # Crea el detector de caras frontal de dlib
        self.predictor = dlib.shape_predictor(predictor_path)  # Carga el modelo de landmarks faciales
        
        # Índices de los puntos de referencia para los ojos
        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
        
        # Umbral para detección de ojos cerrados
        self.EYE_AR_THRESH = 0.20
    
    def detect_face(self, frame):
        """Detecta caras en el frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame a escala de grises
        return self.detector(gray, 0)  # Devuelve la lista de caras detectadas
    
    def get_eye_region(self, frame, landmarks, eye_points):
        """Obtiene la región del ojo a partir de los landmarks faciales."""
        eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) 
                             for point in eye_points], np.int32)  # Extrae las coordenadas de los puntos del ojo
        return eye_region  # Devuelve el polígono que delimita la región del ojo
    
    def eye_aspect_ratio(self, eye_points, landmarks):
        """Calcula la relación de aspecto del ojo (EAR - Eye Aspect Ratio) para detectar parpadeos."""
        try:
            # Obtener las coordenadas de los puntos del ojo
            points = [(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points]
            
            # Calcular las distancias verticales (p2-p6 y p3-p5)
            vert1 = distance.euclidean(points[1], points[5])
            vert2 = distance.euclidean(points[2], points[4])
            
            # Calcular la distancia horizontal (p1-p4)
            horiz = distance.euclidean(points[0], points[3])
            
            # Evitar división por cero
            if horiz == 0:
                return 0.0
                
            # Calcular EAR
            ear = (vert1 + vert2) / (2.0 * horiz)
            
            # Asegurar que el valor esté en un rango razonable
            ear = max(0.0, min(ear, 0.5))
            
            return ear
            
        except Exception as e:
            print(f"Error en eye_aspect_ratio: {e}")
            return 0.0  # En caso de error, asumir ojo cerrado

    def detect_pupil(self, eye_region, frame, eye_side, landmarks=None):
        """Detecta la pupila en la región del ojo."""
        try:
            # Verificar si el ojo está cerrado usando EAR
            if landmarks is not None:
                if eye_side == 'left':
                    ear = self.eye_aspect_ratio(self.LEFT_EYE_POINTS, landmarks)
                else:
                    ear = self.eye_aspect_ratio(self.RIGHT_EYE_POINTS, landmarks)
                
                # Si el ojo está cerrado según el umbral, retornar None
                if ear < self.EYE_AR_THRESH:
                    return None, None

            # Obtener el rectángulo delimitador del ojo
            x, y, w, h = cv2.boundingRect(eye_region)
            
            # Extraer la región del ojo
            eye = frame[y:y+h, x:x+w]
            
            if eye.size == 0 or eye.shape[0] == 0 or eye.shape[1] == 0:
                return None, None
                
            # Convertir a escala de grises
            gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral adaptativo
            thresh = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)  # Resalta la pupila (oscura)
            
            # Operación de apertura para eliminar ruido
            kernel = np.ones((3, 3), np.uint8)  # Crea una matriz de la imagen para operaciones morfológicas
            thresh = cv2.erode(thresh, kernel, iterations=1)  # Erosiona para eliminar ruido blanco de la pupila
            thresh = cv2.dilate(thresh, kernel, iterations=2)  # Dilata para recuperar el tamaño original de la pupila
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Ordenar contornos por área
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Tomar el contorno más grande
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                
                # Filtrar por área mínima
                min_area = max(5, (w * h) * 0.01)
                if area < min_area:
                    return None, None
                    
                # Obtener el círculo que mejor se ajusta al contorno
                (center_x, center_y), _ = cv2.minEnclosingCircle(cnt)
                center_x, center_y = int(center_x), int(center_y)
                
                # Ajustar coordenadas al frame completo
                center_x += x
                center_y += y
                
                # Verificar que la pupila detectada esté dentro de la región del ojo
                if (0 <= center_x < frame.shape[1] and 
                    0 <= center_y < frame.shape[0] and 
                    cv2.pointPolygonTest(eye_region, (center_x, center_y), False) >= 0):
                    return center_x, center_y
                
            return None, None
            
        except Exception as e:
            print(f"Error en detect_pupil: {e}")
            return None, None
            
    def track_pupil(self):
        """Bucle principal para ejecutar el detector de pupilas."""
        import cv2
        from video_capture import VideoCapture
        
        # Inicializar la cámara
        try:
            cap = VideoCapture(camera_index=0, width=640, height=480)
            print("Cámara iniciada correctamente")
        except RuntimeError as e:
            print(f"Error al iniciar la cámara: {e}")
            return
        
        print("Presiona 'q' para salir")
        
        try:
           
          # prev_time = time.time()

           while True:
                # Leer un frame de la cámara
                frame = cap.get_frame()
               # start_time = time.time()
                if frame is None:
                    print("Error: No se pudo capturar el frame.")
                    cv2.destroyAllWindows()
                    sys.exit(1)
                
                # Detectar caras
                faces = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
                
                if len(faces) > 0:
                    # Obtener landmarks faciales
                    landmarks = self.predictor(frame, faces[0])
                    
                    # Obtener regiones de los ojos
                    left_eye_region = self.get_eye_region(frame, landmarks, self.LEFT_EYE_POINTS)
                    right_eye_region = self.get_eye_region(frame, landmarks, self.RIGHT_EYE_POINTS)
                    
                    # Detectar pupilas
                    left_pupil_x, left_pupil_y = self.detect_pupil(left_eye_region, frame, 'left', landmarks)
                    right_pupil_x, right_pupil_y = self.detect_pupil(right_eye_region, frame, 'right', landmarks)
                    
                    if left_pupil_x is not None and left_pupil_y is not None:
                        # Dibujar círculo en la pupila izquierda
                        cv2.circle(frame, (left_pupil_x, left_pupil_y), 3, (0, 0, 255), -1)
                    
                    if right_pupil_x is not None and right_pupil_y is not None:
                        # Dibujar círculo en la pupila derecha
                        cv2.circle(frame, (right_pupil_x, right_pupil_y), 3, (0, 0, 255), -1)
                '''
                # Calcular FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # Calcular latencia
                latency = (time.time() - start_time) * 1000  # en milisegundos
                cv2.putText(frame, f'Latency: {latency:.1f} ms', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                '''
                # Mostrar el frame con las pupilas detectadas
                cv2.namedWindow('Deteccion de Pupilas', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Deteccion de Pupilas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Deteccion de Pupilas', frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nPrograma interrumpido por el usuario")
            
        except Exception as e:
            print(f"Error en track_pupil: {e}")
            
        finally:
            # Liberar recursos
            if 'cap' in locals():
                try:
                    cap.release()
                except Exception as e:
                    print(f"Error al liberar la cámara: {e}")
            cv2.destroyAllWindows()
            print("Recursos liberados")

def main():

    """Función principal para ejecutar el detector de pupilas."""
    try:
        # Inicializar el detector de pupilas
        print("Iniciando detector de pupilas...")
        print("Presiona 'q' para salir del modo pantalla completa")
        pupil_detector = PupilDetector()
        
        # Iniciar el seguimiento de pupilas
        pupil_detector.track_pupil()
        
    except Exception as e:
        print(f"Error en el programa principal: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()