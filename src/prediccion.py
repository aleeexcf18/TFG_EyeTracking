from train import model_left
from train import model_right

def predict_gaze(left_pupil, right_pupil):
    left_pred = model_left.predict([[left_pupil[0], left_pupil[1]]])[0]
    right_pred = model_right.predict([[right_pupil[0], right_pupil[1]]])[0]
    return (int((left_pred[0] + right_pred[0]) / 2), int((left_pred[1] + right_pred[1]) / 2))