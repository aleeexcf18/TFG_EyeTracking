import customtkinter as ctk
import subprocess
import sys
from PIL import Image

def app():
    # Configurar apariencia
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("blue")

    # Crea ventana principal
    root = ctk.CTk()
    root.title("Eye Tracker")
    root.geometry("700x500")

    # Ruta de la imagen original
    entrada = "../utils/eye.png"

    # Imagen de icono en la ventana
    img = Image.open(entrada)
    img_redimensionada = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_redimensionada.save("../utils/eye_icon.ico", format="ICO")
    root.iconbitmap("../utils/eye_icon.ico")

    # Título
    titulo = ctk.CTkLabel(root, text="Eye Tracker", font=ctk.CTkFont(size=22, weight="bold"))
    titulo.pack(pady=20)

    # Funciones para ejecutar scripts
    def ejecutar_script_1():
        subprocess.run([sys.executable, "landmarks.py"])

    def ejecutar_script_2():
        subprocess.run([sys.executable, "face.py"])

    def ejecutar_script_3():
        subprocess.run([sys.executable, "pupil_detector.py"])

    def ejecutar_script_4():
        subprocess.run([sys.executable, "calibrator.py"])

    def ejecutar_script_5():
        subprocess.run([sys.executable, "mouse_controller.py"])

    #Botones
    btn1 = ctk.CTkButton(root, text="Puntos Faciales", font=ctk.CTkFont(size=16), command=ejecutar_script_1, width=220, height=50)
    btn1.pack(pady=10)

    btn2 = ctk.CTkButton(root, text="Deteccion del Rostro", font=ctk.CTkFont(size=16), command=ejecutar_script_2, width=220, height=50)
    btn2.pack(pady=10)

    btn3 = ctk.CTkButton(root, text="Seguimiento de Pupila", font=ctk.CTkFont(size=16), command=ejecutar_script_3, width=220, height=50)
    btn3.pack(pady=10)

    btn4 = ctk.CTkButton(root, text="Calibracion", font=ctk.CTkFont(size=16), command=ejecutar_script_4, width=220, height=50)
    btn4.pack(pady=10)

    btn5 = ctk.CTkButton(root, text="Control del raton", font=ctk.CTkFont(size=16), command=ejecutar_script_5, width=220, height=50)
    btn5.pack(pady=10)

    return root

if __name__ == "__main__":
    app = app()
    app.mainloop()