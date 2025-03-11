import tkinter as tk
import subprocess
import sys

def ejecutar_script_1():
    subprocess.run([sys.executable, "C:\\Users\\anton\\Downloads\\prueba.py"])

#def ejecutar_script_2():
#    subprocess.run([sys.executable, "script2.py"])

# Crear ventana
root = tk.Tk()
root.title("Camara jiji")
root.geometry("1080x720")

# Botones
btn1 = tk.Button(root, text="Seguimiento pupila", command=ejecutar_script_1)
btn1.pack(pady=10)

#btn2 = tk.Button(root, text="Ejecutar Script 2", command=ejecutar_script_2)
#btn2.pack(pady=10)

# Iniciar interfaz
root.mainloop()