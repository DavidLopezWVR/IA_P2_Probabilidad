#21110344  David López Rojas  6E2

import tensorflow as tf  # Importar la biblioteca TensorFlow para crear y entrenar modelos de aprendizaje profundo
from tensorflow.keras import layers, models  # Importar las capas y modelos de Keras para construir redes neuronales

# Definir la arquitectura de la red neuronal artificial (ANN)
model = models.Sequential([  # Inicializar un modelo secuencial
    layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: aplanar la imagen de 28x28 píxeles a un vector de 784 elementos (28*28)
    
    layers.Dense(128, activation='relu'),  # Capa oculta: 128 neuronas con función de activación ReLU, permite modelar relaciones no lineales
    
    layers.Dense(10, activation='softmax')  # Capa de salida: 10 neuronas (una para cada clase de dígitos) con activación softmax para producir probabilidades
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador Adam para actualizar los pesos del modelo
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación multiclase (se usa con etiquetas enteras)
              metrics=['accuracy'])  # Métrica a evaluar durante el entrenamiento

# Cargar y preprocesar datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist  # Cargar el conjunto de datos MNIST de Keras, que contiene imágenes de dígitos escritos a mano
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar los datos dividiendo entre 255 para que los valores estén en el rango [0, 1]

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5)  # Entrenar el modelo en el conjunto de entrenamiento durante 5 épocas

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluar el modelo en el conjunto de prueba y obtener la pérdida y precisión
print("Precisión del modelo en el conjunto de prueba:", test_acc)  # Imprimir la precisión del modelo en el conjunto de prueba
