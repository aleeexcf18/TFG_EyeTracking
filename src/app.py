import tkinter as tk
import subprocess
import sys

def ejecutar_script_1():
    subprocess.run([sys.executable, "src\\ale.py"])

def ejecutar_script_2():
    subprocess.run([sys.executable, "src\\calibration.py"])

def ejecutar_script_3():
    subprocess.run([sys.executable, "src\\face.py"])

def ejecutar_script_4():
    subprocess.run([sys.executable, "src\\landmarks.py"])

# Crear ventana
root = tk.Tk()
root.title("Aplicación")
root.geometry("1080x720")

# Botones
btn1 = tk.Button(root, text="Seguimiento pupila", command=ejecutar_script_1)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Calibración", command=ejecutar_script_2)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="Detección del rostro", command=ejecutar_script_3)
btn3.pack(pady=10)

btn4 = tk.Button(root, text="Puntos faciales", command=ejecutar_script_4)
btn4.pack(pady=10)

# Iniciar interfaz
root.mainloop()