import cv2
from tracker import EyeTracker, EyeGazeEstimator
from calibration import Calibrator

def main():
    cap = cv2.VideoCapture(0)

    calibrator = Calibrator()
    calibrator.calibrate(cap)

    eye_tracker = EyeTracker()
    gaze_estimator = EyeGazeEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eye_tracker.process_frame(frame)
        gaze_direction = gaze_estimator.estimate_gaze(eye_tracker.detect_landmarks(frame), frame)

        if gaze_direction is not None:
            # Mostrar la dirección de la mirada (esto puede ser un vector en 3D)
            print(f"Dirección de la mirada: {gaze_direction}")

        cv2.imshow('Eye Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
