#21110344  David López Rojas  6E2

# Aprendizaje Profundo (Deep Learning)        

import tensorflow as tf  # Importar la biblioteca TensorFlow para crear y entrenar modelos de aprendizaje profundo
from tensorflow.keras import layers, models  # Importar las capas y modelos de Keras para construir redes neuronales

# Definir la arquitectura de la red neuronal convolucional (CNN)
model = models.Sequential([  # Inicializar un modelo secuencial
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
    # Capa convolucional con 32 filtros de tamaño 3x3, función de activación ReLU, y forma de entrada de imágenes en escala de grises (28x28)
    
    layers.MaxPooling2D((2, 2)),  # Capa de max pooling para reducir la dimensionalidad a la mitad (2x2)
    
    layers.Conv2D(64, (3, 3), activation='relu'),  # Segunda capa convolucional con 64 filtros de tamaño 3x3 y activación ReLU
    
    layers.MaxPooling2D((2, 2)),  # Otra capa de max pooling para continuar reduciendo la dimensionalidad
    
    layers.Conv2D(64, (3, 3), activation='relu'),  # Tercera capa convolucional con 64 filtros de tamaño 3x3 y activación ReLU
    
    layers.Flatten(),  # Aplanar la salida de las capas convolucionales para conectarla a capas densas
    
    layers.Dense(64, activation='relu'),  # Capa densa completamente conectada con 64 neuronas y activación ReLU
    
    layers.Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada clase) y activación softmax para clasificación multiclase
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador Adam para actualizar los pesos del modelo
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación multiclase
              metrics=['accuracy'])  # Métrica a evaluar durante el entrenamiento

# Cargar y preprocesar datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist  # Cargar el conjunto de datos MNIST de Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar los datos dividiendo entre 255 para que estén en el rango [0, 1]

# Agregar una dimensión al conjunto de datos para que sea compatible con la entrada de la CNN
x_train = x_train[..., tf.newaxis]  # Agregar una nueva dimensión para representar el canal de color (1 para imágenes en escala de grises)
x_test = x_test[..., tf.newaxis]  # Hacer lo mismo para el conjunto de prueba

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))  
# Entrenar el modelo en el conjunto de entrenamiento durante 5 épocas y validar en el conjunto de prueba

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)  # Evaluar el modelo en el conjunto de prueba y obtener la pérdida y precisión
print("Precisión del modelo en el conjunto de prueba:", test_acc)  # Imprimir la precisión del modelo en el conjunto de prueba
