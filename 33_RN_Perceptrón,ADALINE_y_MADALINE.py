#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para manejar arreglos y operaciones matemáticas

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate  # Establecer la tasa de aprendizaje
        self.n_iterations = n_iterations      # Establecer el número de iteraciones para el entrenamiento

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])  # Inicializar los pesos con ceros, incluyendo el bias
        self.errors = []                          # Lista para almacenar el número de errores en cada iteración

        for _ in range(self.n_iterations):        # Iterar por el número de iteraciones definido
            errors = 0                            # Inicializar el contador de errores en cada iteración
            for xi, target in zip(X, y):         # Iterar sobre cada muestra y su correspondiente etiqueta
                update = self.learning_rate * (target - self.predict(xi))  # Calcular la actualización de los pesos
                self.weights[1:] += update * xi   # Actualizar los pesos de las características
                self.weights[0] += update          # Actualizar el bias
                errors += int(update != 0.0)      # Contar errores si la actualización no es cero
            self.errors.append(errors)            # Almacenar el número de errores en la lista
        return self  # Retornar la instancia del modelo

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]  # Calcular la entrada neta como el producto escalar más el bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # Predecir la clase como 1 o -1 según la entrada neta

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate  # Establecer la tasa de aprendizaje
        self.n_iterations = n_iterations      # Establecer el número de iteraciones para el entrenamiento

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])  # Inicializar los pesos con ceros, incluyendo el bias
        self.cost = []                           # Lista para almacenar el costo en cada iteración

        for _ in range(self.n_iterations):       # Iterar por el número de iteraciones definido
            output = self.activation(X)          # Calcular la salida usando la activación
            errors = (y - output)                # Calcular el error entre la etiqueta y la salida
            self.weights[1:] += self.learning_rate * X.T.dot(errors)  # Actualizar los pesos
            self.weights[0] += self.learning_rate * errors.sum()       # Actualizar el bias
            cost = (errors**2).sum() / 2.0       # Calcular el costo (función de pérdida)
            self.cost.append(cost)                # Almacenar el costo en la lista
        return self  # Retornar la instancia del modelo

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]  # Calcular la entrada neta

    def activation(self, X):
        return self.net_input(X)  # Retornar la entrada neta como la activación

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)  # Predecir la clase como 1 o -1

class Madaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate  # Establecer la tasa de aprendizaje
        self.n_iterations = n_iterations      # Establecer el número de iteraciones para el entrenamiento

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])  # Inicializar los pesos con ceros, incluyendo el bias
        self.errors = []                          # Lista para almacenar el número de errores en cada iteración

        for _ in range(self.n_iterations):        # Iterar por el número de iteraciones definido
            errors = 0                            # Inicializar el contador de errores en cada iteración
            for xi, target in zip(X, y):         # Iterar sobre cada muestra y su correspondiente etiqueta
                update = self.learning_rate * (target - self.predict(xi))  # Calcular la actualización de los pesos
                self.weights[1:] += update * xi   # Actualizar los pesos de las características
                self.weights[0] += update          # Actualizar el bias
                errors += int(update != 0.0)      # Contar errores si la actualización no es cero
            self.errors.append(errors)            # Almacenar el número de errores en la lista
        return self  # Retornar la instancia del modelo

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]  # Calcular la entrada neta

    def activation(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # Aplicar la activación, retornando 1 o -1

    def predict(self, X):
        return self.activation(X)  # Retornar la activación como la predicción

# Ejemplo de uso
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Definir un conjunto de datos de ejemplo
y = np.array([-1, -1, 1, 1])                     # Definir las etiquetas correspondientes

# Perceptron
perceptron = Perceptron()  # Instanciar el modelo Perceptron
perceptron.fit(X, y)       # Ajustar el modelo a los datos de entrenamiento
print("Perceptron weights:", perceptron.weights)  # Imprimir los pesos del Perceptron

# ADALINE
adaline = Adaline()  # Instanciar el modelo ADALINE
adaline.fit(X, y)    # Ajustar el modelo a los datos de entrenamiento
print("ADALINE weights:", adaline.weights)  # Imprimir los pesos del ADALINE

# MADALINE
madaline = Madaline()  # Instanciar el modelo MADALINE
madaline.fit(X, y)     # Ajustar el modelo a los datos de entrenamiento
print("MADALINE weights:", madaline.weights)  # Imprimir los pesos del MADALINE
