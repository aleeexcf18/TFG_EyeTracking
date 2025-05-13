import customtkinter as ctk
import subprocess
import sys
from PIL import Image

# Configurar apariencia
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Crear ventana principal
root = ctk.CTk()
root.title("Eye Tracker")
root.geometry("600x400")

# Ruta de la imagen original
entrada = "assets/eye.png"  # Cambia esto si es necesario
salida = "assets/eye_icon.png"  # Imagen de salida redimensionada

# Imagen de icono en la ventana
img = Image.open(entrada)
img_redimensionada = img.resize((64, 64), Image.Resampling.LANCZOS)
img_redimensionada.save("assets/eye_icon.ico", format="ICO")
root.iconbitmap("assets/eye_icon.ico")

# Título
titulo = ctk.CTkLabel(root, text="Eye Tracker", font=ctk.CTkFont(size=22, weight="bold"))
titulo.pack(pady=20)

# Funciones para ejecutar scripts
def ejecutar_script_1():
    subprocess.run([sys.executable, "src\\ale.py"])

def ejecutar_script_2():
    subprocess.run([sys.executable, "src\\calibration.py"])

def ejecutar_script_3():
    subprocess.run([sys.executable, "src\\face.py"])

def ejecutar_script_4():
    subprocess.run([sys.executable, "src\\landmarks.py"])

# Botones modernos
btn1 = ctk.CTkButton(root, text="Seguimiento de Pupila", font=ctk.CTkFont(size=16), command=ejecutar_script_1, width=220, height=50)
btn1.pack(pady=10)

btn2 = ctk.CTkButton(root, text="Calibración", font=ctk.CTkFont(size=16), command=ejecutar_script_2, width=220, height=50)
btn2.pack(pady=10)

btn3 = ctk.CTkButton(root, text="Detección del Rostro", font=ctk.CTkFont(size=16), command=ejecutar_script_3, width=220, height=50)
btn3.pack(pady=10)

btn4 = ctk.CTkButton(root, text="Puntos Faciales", font=ctk.CTkFont(size=16), command=ejecutar_script_4, width=220, height=50)
btn4.pack(pady=10)

# Iniciar app
root.mainloop()