#21110344  David López Rojas  6e2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes
import numpy as np  # Importar NumPy para operaciones con matrices y arreglos
from sklearn.model_selection import train_test_split  # Importar función para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.svm import SVC  # Importar el clasificador de máquinas de soporte vectorial
from sklearn.metrics import accuracy_score  # Importar la función para calcular la precisión del modelo

# Función para extraer características de una imagen (en este caso, solo se usa la intensidad de los píxeles)
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
    return gray_image.flatten()  # Aplanar la imagen en un vector unidimensional como característica

# Cargar imágenes de líneas etiquetadas y no etiquetadas
labeled_lines = []  # Lista para almacenar líneas etiquetadas
unlabeled_lines = []  # Lista para almacenar líneas no etiquetadas

# Cargar imágenes de líneas etiquetadas
for i in range(1, 101):  # Bucle para cargar 100 imágenes etiquetadas
    image = cv2.imread(f'labeled_lines/line_{i}.png')  # Leer la imagen de la carpeta de líneas etiquetadas
    labeled_lines.append((image, 1))  # Añadir la imagen y la etiqueta 1 (etiquetada)

# Cargar imágenes de líneas no etiquetadas
for i in range(1, 101):  # Bucle para cargar 100 imágenes no etiquetadas
    image = cv2.imread(f'unlabeled_lines/line_{i}.png')  # Leer la imagen de la carpeta de líneas no etiquetadas
    unlabeled_lines.append((image, 0))  # Añadir la imagen y la etiqueta 0 (no etiquetada)

# Combinar líneas etiquetadas y no etiquetadas y dividirlas en conjuntos de entrenamiento y prueba
data = labeled_lines + unlabeled_lines  # Combinar ambas listas en una sola
X = [extract_features(image) for image, _ in data]  # Extraer características de todas las imágenes
y = [label for _, label in data]  # Extraer las etiquetas correspondientes

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de clasificación (SVM en este caso)
svm_model = SVC(kernel='linear')  # Inicializar el clasificador SVM con un kernel lineal
svm_model.fit(X_train, y_train)  # Entrenar el modelo con los datos de entrenamiento

# Predecir etiquetas para el conjunto de prueba
y_pred = svm_model.predict(X_test)  # Realizar predicciones en el conjunto de prueba

# Calcular precisión del modelo
accuracy = accuracy_score(y_test, y_pred)  # Calcular la precisión comparando las predicciones con las etiquetas verdaderas
print("Precisión del modelo:", accuracy)  # Imprimir la precisión del modelo
