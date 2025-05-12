import cv2
import numpy as np

def detect_pupil(eye_crop):
    gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

def process_eye(frame, landmarks, eye_indices):
    x1, y1 = landmarks.part(eye_indices[0]).x, landmarks.part(eye_indices[0]).y
    x2, y2 = landmarks.part(eye_indices[3]).x, landmarks.part(eye_indices[3]).y
    eye_crop = frame[y1:y2, x1:x2]
    
    if eye_crop.size == 0:
        return None
    
    pupil = detect_pupil(eye_crop)
    if pupil:
        return (pupil[0] + x1, pupil[1] + y1)
    return None