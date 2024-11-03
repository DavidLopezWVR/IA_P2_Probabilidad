#21110344  David López Rojas  6E2

import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons  # Número de neuronas en la red de Hopfield
        self.weights = np.zeros((num_neurons, num_neurons))  # Inicializar la matriz de pesos con ceros

    def train(self, patterns):
        # Entrenar la red de Hopfield utilizando un conjunto de patrones
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Asegurar que el patrón tenga la forma de columna
            self.weights += np.dot(pattern, pattern.T)  # Actualizar la matriz de pesos sumando el producto externo
            np.fill_diagonal(self.weights, 0)  # Establecer la diagonal de la matriz de pesos en cero (sin auto-conexiones)

    def predict(self, pattern, max_iterations=100):
        # Predecir el patrón más cercano a partir de un patrón ruidoso
        pattern = pattern.reshape(-1, 1)  # Asegurar que el patrón tenga la forma de columna
        for _ in range(max_iterations):
            # Calcular el nuevo patrón utilizando la función de activación (signo de la suma ponderada)
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):  # Si no hay cambios en el patrón, retornar el patrón actual
                return new_pattern
            pattern = new_pattern  # Actualizar el patrón para la siguiente iteración
        return pattern  # Retornar el patrón final después de las iteraciones

# Ejemplo de uso
patterns = np.array([[1, 1, -1, -1],  # Patrones de entrenamiento
                     [-1, -1, 1, 1],
                     [1, -1, 1, -1]])

hopfield = HopfieldNetwork(num_neurons=len(patterns[0]))  # Crear una red de Hopfield con el número adecuado de neuronas
hopfield.train(patterns)  # Entrenar la red con los patrones proporcionados

# Introducimos un patrón ruidoso para recuperar el patrón original
noisy_pattern = np.array([-1, -1, 1, -1])  # Patrón ruidoso que se asemeja a uno de los patrones originales

retrieved_pattern = hopfield.predict(noisy_pattern)  # Intentar recuperar el patrón original a partir del ruidoso
print("Patrón original recuperado:", retrieved_pattern.flatten())  # Imprimir el patrón recuperado
