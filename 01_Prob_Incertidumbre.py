#21110344  David López Rojas  6E2

import numpy as np

# Función de activación sigmoid
def sigmoid(x):
    # Calcula la función sigmoide, que transforma valores reales en el rango (0, 1)
    return 1 / (1 + np.exp(-x))

# Definición de una red neuronal simple
class NeuralNetwork:
    def __init__(self):
        # Inicializa los parámetros de la red neuronal: pesos y sesgo
        self.weights = np.random.randn(2, 1)  # Pesos aleatorios para 2 entradas
        self.bias = np.random.randn(1)        # Sesgo aleatorio

    # Método para predecir la salida dada una entrada
    def predict(self, x):
        # Calcula la predicción mediante el producto punto de las entradas y los pesos, más el sesgo
        return sigmoid(np.dot(x, self.weights) + self.bias)

# Función principal del script
def main():
    # Datos de entrada: todas las combinaciones posibles de dos valores binarios (0 o 1)
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Crear una instancia de la red neuronal
    neural_network = NeuralNetwork()

    # Realizar predicciones para cada punto de entrada
    for i, input_point in enumerate(input_data):
        # Obtener la predicción de la red neuronal para el punto de entrada
        prediction = neural_network.predict(input_point)
        
        # Agregar incertidumbre a la predicción: añadiendo ruido gaussiano
        noisy_prediction = prediction + np.random.normal(scale=0.1)  # Ruido con desviación estándar de 0.1
        
        # Imprimir la entrada, la predicción y la predicción con incertidumbre
        print(f"Entrada: {input_point}, Predicción: {prediction}, Predicción con Incertidumbre: {noisy_prediction}")

# Punto de entrada del programa
if __name__ == "__main__":
    main()
