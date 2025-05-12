import numpy as np
from calibration import calibration_data
from sklearn.linear_model import LinearRegression

# Modelos de regresión
model_left = LinearRegression()
model_right = LinearRegression()

def train_models():
    # Convertir los datos de calibración de manera más eficiente
    X_left = np.array([[lp[0], lp[1]] for lp, _, _ in calibration_data])
    X_right = np.array([[rp[0], rp[1]] for _, rp, _ in calibration_data])
    y = np.array([point for _, _, point in calibration_data])
    model_left.fit(X_left, y)
    model_right.fit(X_right, y)
    print("Modelos entrenados.")