import cv2
from captura import cap, detector, predictor
from procesamiento import process_eye
from calibration import calibrate
from prediccion import predict_gaze

# Configuración de ventana
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def main_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        flipped_frame = cv2.flip(frame, 1)
        for face in faces:
            landmarks = predictor(gray, face)
            left_pupil = process_eye(frame, landmarks, [36, 37, 38, 39, 40, 41])
            right_pupil = process_eye(frame, landmarks, [42, 43, 44, 45, 46, 47])
            if left_pupil and right_pupil:
                gaze_x, gaze_y = predict_gaze(left_pupil, right_pupil)
                cv2.circle(flipped_frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
                cv2.putText(flipped_frame, f"Mirada: ({gaze_x}, {gaze_y})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", flipped_frame)
        if cv2.waitKey(1) & 0xFF == 13:
            break

if __name__ == "__main__":
    calibrate()
    main_loop()
    cap.release()
    cv2.destroyAllWindows()