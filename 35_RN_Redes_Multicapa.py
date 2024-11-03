#21110344  David López Rojas  6E2

import tensorflow as tf
from tensorflow.keras import layers, models

# Definir la arquitectura de la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: convierte las imágenes 28x28 en un vector unidimensional
    layers.Dense(128, activation='relu'),  # Capa oculta 1: 128 neuronas que utilizan la función de activación ReLU (Rectified Linear Unit)
    layers.Dense(64, activation='relu'),   # Capa oculta 2: 64 neuronas, también con función de activación ReLU
    layers.Dense(10, activation='softmax') # Capa de salida: 10 neuronas para clasificar en 10 clases (dígitos del 0 al 9) usando la función de activación Softmax
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador Adam, que adapta la tasa de aprendizaje durante el entrenamiento
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación multiclase
              metrics=['accuracy'])  # Métrica de evaluación, en este caso precisión

# Entrenar el modelo con datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist  # Cargar el conjunto de datos MNIST, que contiene imágenes de dígitos
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar los datos a un rango de [0, 1] dividiendo por 255

# Ajustar el modelo
model.fit(x_train, y_train, epochs=5)  # Entrenar el modelo durante 5 épocas con los datos de entrenamiento

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluar el rendimiento del modelo en el conjunto de prueba
print("Precisión del modelo en el conjunto de prueba:", test_acc)  # Imprimir la precisión del modelo
