"""
# Hand Washing Technique Classifier using MediaPipe and TensorFlow

This Python script trains a neural network to classify correct vs incorrect hand washing techniques 
using hand landmarks extracted with MediaPipe.

## Description
The program:
1. Processes images of hand washing steps (correct and incorrect techniques)
2. Extracts hand landmarks using MediaPipe
3. Trains a neural network to classify correct technique
4. Evaluates model performance

## Author
[Ricardo de Jesús Cepeda Varela]
[ricky.cepeda01@gmail.com]

## Date
[04/20/2025]

## Usage
1. Upload hand washing images to Google Drive in the specified folder structure
2. Run in Google Colab with GPU acceleration
3. Model will be saved as 'handwash_model.h5'

## Dependencies
- Python 3.x
- TensorFlow 2.x
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
"""
#///////////////////////////////////////////////////CELDA 1: INSTALACIÓN Y ACTUALIZACIÓN DE LIBRERÍAS /////////////////////////////////////////////////////////////////////
!pip install -q mediapipe
!wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
!pip install --upgrade tensorflow

#///////////////////////////////////////////////////CELDA 2: SELECCIÓN DE DATOS DE ENTRENAMIENTO Y PROCESAMIENTO CON MEDIAPIPE/////////////////////////////////////////////////////////////////////
import tensorflow as tf
import numpy as np
import cv2
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from google.colab import drive
#Montamos el dive para leer los contenidos de la base de datos
drive.mount('/content/drive')

#importamos las funciones de mediapipe para la deteccion de manos y el trazado de dibujos sobre las manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#definimos una función para extraer la información de los landmarks que señala mediapipe
def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)    #se lee la ruta de los datos
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    results = hands.process(image_rgb)
    hands.close()
# se guardan los datos de los landmarks en arrays de 3 dimensiones xyz
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None  # Devuelve None si no se detecta mano y se excluye del entrenamiento


#definimos una funcion para cargar los datos correctos e incorrectos para el entrenamiento
def load_hand_data(correct_folder, incorrect_folders):
    X, y = [], []

    # Cargar imágenes correctas (Step1)
    correct_images = [f for f in os.listdir(correct_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for filename in correct_images:
        img_path = os.path.join(correct_folder, filename)
        landmarks = extract_hand_landmarks(img_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(1)  # Etiqueta para imágenes correctas

    # Cargar imágenes incorrectas (Step2 - Step7)
    for folder in incorrect_folders:
        if os.path.exists(folder):
            incorrect_images = [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            for filename in incorrect_images:
                img_path = os.path.join(folder, filename)
                landmarks = extract_hand_landmarks(img_path)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(0)  # Etiqueta para imágenes incorrectas

    return np.array(X), np.array(y)
#agregamos las rutas de los datos
correct_data_folder = "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step1"
incorrect_data_folders = [
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step2",
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step3",
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step4",
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step5",
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step6",
    "/content/drive/MyDrive/PRACTICAS LAVADO DE MANOS/Frames/Step7"
]

X, y = load_hand_data(correct_data_folder, incorrect_data_folders)

# Verifica que haya datos antes de dividir
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    print("No se detectaron manos en ninguna imagen.")

#/////////////////////////////////////////////////////CELDA 3: DEFINICIÓN DE ARQUITECTURA DE RED NEURONAL Y ENTRENAMIENTO///////////////////////////////////////////////////////////////////
# Verificar que haya datos cargados antes de entrenar
if len(X_train) > 0:
    # Definir el modelo de red neuronal
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')  # Dos clases: correcta (1) e incorrecta (0)
    ])

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test))

    # Guardar el modelo entrenado
    model.save("handwash_model.h5")
else:
    print("No hay datos suficientes para entrenar el modelo.")

#/////////////////////////////////////////////////////CELDA 4: VALIDACIÓN DEL RENDIMIENTO DEL ENTRENAMIENTO ÉPOCA VS PRECISIÓN///////////////////////////////////////////////////////////////////
import matplotlib.pyplot as plt # import matplotlib and assign it to the alias plt

# Plot validation and testing accuracy
plt.figure(figsize=(12,6)) # Se establece el tamaño de la gráfica.
plt.plot(history.history['accuracy']) # Se gráfica la exactitud del modelo a través de las épocas.
plt.plot(history.history['val_accuracy']) # Se gráfica la exactitud del modelo con los datos de validación del entrenamiento a través de las épocas.
plt.xlabel('epoch') # El eje x corresponde a las épocas.
plt.ylabel('accuracy') # El eje y corresponde a las exactitud.
plt.legend(['train', 'val']) # Se añade una leyenda para distinguir entre entrenamiento y validación.
plt.grid(); # Se añade una cuadrícula a la gráfica para mejorar la legibilidad.

#/////////////////////////////////////////////////////CELDA 5: VALIDACIÓN DEL RENDIMIENTO DEL ENTRENAMIENTO ÉPOCA VS PÉRDIDA///////////////////////////////////////////////////////////////////

# Plot validation and testing loss
plt.figure(figsize=(12,6)) # Se establece el tamaño de la gráfica.
plt.plot(history.history['loss']) # Se gráfica la pérdida del conjunto de entrenamiento a lo largo de las épocas.
plt.plot(history.history['val_loss']) # Se gráfica la pérdida con los datos de validación del entrenamiento a lo largo de las épocas.
plt.xlabel('epoch') # El eje x corresponde a las épocas.
plt.ylabel('loss') # El eje y corresponde a la pérdida.
plt.legend(['train', 'val'])  # Se añade una leyenda para distinguir entre entrenamiento y validación.
plt.grid(); # Se añade una cuadrícula a la gráfica para mejorar la legibilidad.
