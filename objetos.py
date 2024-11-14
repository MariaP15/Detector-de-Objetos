import torch
import cv2
from collections import Counter
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Cargar el modelo preentrenado YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Diccionario de traducción de etiquetas
label_translation = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'coche',
    'motorcycle': 'motocicleta',
    'airplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'barco',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'señal de alto',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pájaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa',
    'backpack': 'mochila',
    'umbrella': 'paraguas',
    'handbag': 'bolso de mano',
    'tie': 'corbata',
    'suitcase': 'maleta',
    'frisbee': 'frisbee',
    'skis': 'esquís',
    'snowboard': 'tabla de snowboard',
    'sports ball': 'pelota de deportes',
    'kite': 'cometa',
    'baseball bat': 'bate de béisbol',
    'baseball glove': 'guante de béisbol',
    'skateboard': 'monopatín',
    'surfboard': 'tabla de surf',
    'tennis racket': 'raqueta de tenis',
    'bottle': 'botella',
    'wine glass': 'copa de vino',
    'cup': 'taza',
    'fork': 'tenedor',
    'knife': 'cuchillo',
    'spoon': 'cuchara',
    'bowl': 'tazón',
    'banana': 'plátano',
    'apple': 'manzana',
    'sandwich': 'sándwich',
    'orange': 'naranja',
    'broccoli': 'brócoli',
    'carrot': 'zanahoria',
    'hot dog': 'perrito caliente',
    'pizza': 'pizza',
    'donut': 'dona',
    'cake': 'pastel',
    'chair': 'silla',
    'couch': 'sofá',
    'potted plant': 'planta en maceta',
    'bed': 'cama',
    'dining table': 'mesa de comedor',
    'toilet': 'inodoro',
    'tv': 'televisión',
    'laptop': 'portátil',
    'mouse': 'ratón',
    'remote': 'control remoto',
    'keyboard': 'teclado',
    'cell phone': 'teléfono móvil',
    'microwave': 'microondas',
    'oven': 'horno',
    'toaster': 'tostadora',
    'sink': 'fregadero',
    'refrigerator': 'refrigerador',
    'book': 'libro',
    'clock': 'reloj',
    'vase': 'jarrón',
    'scissors': 'tijeras',
    'teddy bear': 'oso de peluche',
    'hair drier': 'secador de pelo',
    'toothbrush': 'cepillo de dientes'
}

# Lista de rutas de imágenes
image_paths = ['barco1.jpg', 'osos.webp',  'sala2.jpg','umbrellas.jpg', 'personas2.jpg', 'vehiculos.jpg', 'ciudad.jpg', 'sala.webp']
current_image_index = 0  # Índice para la imagen actual

def update_image():
    global current_image_index

    # Cargar la nueva imagen
    img_path = image_paths[current_image_index]
    img = cv2.imread(img_path)

    # Hacer predicciones
    results = model(img)

    # Extraer las detecciones
    detections = results.pred[0]

    # Dibujar los rectángulos (sin texto encima y con grosor más delgado)
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Grosor cambiado a 1

    img_resized = cv2.resize(img, (700, 400))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk
    image_label.place(x='600', y='200')

    # Crear el mensaje de conteo de objetos con pluralización
    labels = [results.names[int(detection[-1])] for detection in detections]
    label_counts = Counter(labels)
    
    # Ordenar los objetos alfabéticamente
    sorted_labels = sorted(label_counts.items(), key=lambda x: label_translation.get(x[0], x[0]))
    
    message = ""
    messagetitle = "Se detectaron:"
    for label, count in sorted_labels:
        label_spanish = label_translation.get(label, label)

        # Añadir plural si el conteo es mayor a 1
        if count > 1:
            label_spanish += 's'

        message += f" {count} {label_spanish}.\n"

    num_objects = len(detections)
    message += f"\nObjetos en total: {num_objects}."

    # Mostrar el mensaje en la ventana hija después de un retraso de 1 segundo
    message_window.after(1000, lambda: message_label.config(text=message))
    message_look.config(text=messagetitle)

    current_image_index = (current_image_index + 1) % len(image_paths)

def on_key_press(event):
    if event.char.lower() == 'c':  # Detecta si se presiona la tecla 'C' o 'c'
        update_image()

# Crear la interfaz con Tkinter
root = tk.Tk()
root.title("Conteo de Objetos Detectados")
root.state("zoomed")
root.configure(bg="White")

# Crear una ventana hija donde se mostrarán los mensajes
message_window = tk.Toplevel(root)
message_window.title("Mensajes de Detección")
message_window.geometry("300x200")
message_window.configure(bg="White")  

# Mostrar el mensaje en la ventana hija
message_look = Label(root, text="", font=("Helvetica", 18, "italic"), justify="center", bg='White')
message_label = Label(root, text="", font=("Helvetica", 16, "italic"), justify="left", bg='White')
message_label.place(x="245", y="270")
message_look.place(x='230', y='200')

title_label = Label(root, text="Objetos identificados en la imagen", font=("Helvetica", 20, "bold"), bg='White')
title_label.pack(pady=8)

image_label = Label(root)
image_label.pack(pady=20)

root.bind('<Key>', on_key_press)

update_image()

root.mainloop()
