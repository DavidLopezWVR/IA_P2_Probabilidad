#21110344  David López Rojas  6E2

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar los pesos de las capas ocultas y de salida con valores aleatorios
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)  # Pesos entre la entrada y la capa oculta
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) # Pesos entre la capa oculta y la capa de salida
        
        # Inicializar los sesgos de las capas ocultas y de salida con valores aleatorios
        self.bias_hidden = np.random.rand(hidden_size)  # Sesgos de la capa oculta
        self.bias_output = np.random.rand(output_size)  # Sesgos de la capa de salida
        
    def sigmoid(self, x):
        # Función de activación Sigmoide
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide, usada para la retropropagación
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        # Propagación hacia adelante: calcular la salida de la red
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)  # Salida de la capa oculta
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)  # Salida final
    
    def backward_propagation(self, inputs, targets, learning_rate):
        # Retropropagación: actualizar los pesos y sesgos basados en el error de la salida
        output_error = targets - self.output  # Error en la capa de salida
        output_delta = output_error * self.sigmoid_derivative(self.output)  # Ajuste de la salida
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)  # Error en la capa oculta
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)  # Ajuste de la capa oculta
        
        # Actualizar los pesos de la capa de salida y de la capa oculta
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        
        # Actualizar los sesgos de la capa de salida y de la capa oculta
        self.bias_output += learning_rate * np.sum(output_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta)
    
    def train(self, inputs, targets, epochs, learning_rate):
        # Entrenar la red neuronal durante un número específico de épocas
        for epoch in range(epochs):
            # Realizar la propagación hacia adelante
            self.forward_propagation(inputs)
            
            # Realizar la retropropagación del error
            self.backward_propagation(inputs, targets, learning_rate)
            
            # Calcular y mostrar el error medio absoluto en cada época
            error = np.mean(np.abs(targets - self.output))
            print(f'Epoch {epoch + 1}/{epochs}, Error: {error}')

# Ejemplo de uso
# Datos de entrada y salida para la función XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])  # Salidas deseadas para la función XOR

# Crear una red neuronal con 2 neuronas de entrada, 2 ocultas y 1 de salida
network = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Entrenar la red neuronal
network.train(inputs, targets, epochs=10000, learning_rate=0.1)  # Entrenar por 10,000 épocas
