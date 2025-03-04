import cv2

def mostrar_mensaje(frame, texto, posicion, font, font_scale, color, grosor):
    cv2.putText(frame, texto, posicion, font, font_scale, color, grosor)

def dibujar_punto(frame, punto, color=(0, 0, 255)):
    cv2.circle(frame, punto, 10, color, -1)