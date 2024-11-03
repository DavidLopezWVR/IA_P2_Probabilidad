#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas y manejo de arreglos

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Calcular la función sigmoide, que comprime la entrada a un rango entre 0 y 1

# Función de activación ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)  # Calcular la función ReLU, que devuelve 0 para valores negativos y el valor mismo para positivos

# Función de activación Tangente Hiperbólica
def tanh(x):
    return np.tanh(x)  # Calcular la tangente hiperbólica, que comprime la entrada a un rango entre -1 y 1

# Función de activación Softmax
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Calcular el exponencial de cada elemento, evitando el desbordamiento restando el máximo
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalizar para obtener probabilidades que suman 1

# Datos de ejemplo
x = np.array([[1, 2, 3],  # Matriz 2x3 con ejemplos de entrada
              [-1, 0, 1]])

# Aplicar las funciones de activación a los datos de ejemplo
sigmoid_output = sigmoid(x)  # Aplicar la función sigmoide a los datos de entrada
relu_output = relu(x)        # Aplicar la función ReLU a los datos de entrada
tanh_output = tanh(x)        # Aplicar la tangente hiperbólica a los datos de entrada
softmax_output = softmax(x)  # Aplicar la función softmax a los datos de entrada

# Mostrar los resultados
print("Función de activación Sigmoide:")  # Imprimir encabezado para la salida de la sigmoide
print(sigmoid_output)  # Imprimir los resultados de la función sigmoide

print("\nFunción de activación ReLU:")  # Imprimir encabezado para la salida de ReLU
print(relu_output)  # Imprimir los resultados de la función ReLU

print("\nFunción de activación Tangente Hiperbólica:")  # Imprimir encabezado para la salida de tanh
print(tanh_output)  # Imprimir los resultados de la tangente hiperbólica

print("\nFunción de activación Softmax:")  # Imprimir encabezado para la salida de softmax
print(softmax_output)  # Imprimir los resultados de la función softmax
