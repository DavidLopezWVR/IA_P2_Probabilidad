#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar resultados

# Definir la clase para implementar el Filtro de Kalman
class KalmanFilter:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A = A  # Matriz de transición de estado, describe cómo evoluciona el estado entre pasos
        self.H = H  # Matriz de observación, relaciona el estado con las mediciones
        self.Q = Q  # Covarianza del proceso, representa la incertidumbre en el modelo de estado
        self.R = R  # Covarianza de la medición, representa la incertidumbre en las mediciones
        self.x = x0  # Estado inicial
        self.P = P0  # Covarianza inicial del estado

    # Método para predecir el siguiente estado
    def predict(self):
        self.x = np.dot(self.A, self.x)  # Actualizar el estado predicho
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # Actualizar la covarianza del estado

    # Método para actualizar el estado basado en una nueva medición
    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Calcular la innovación (diferencia entre medición y predicción)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Calcular la covarianza de la innovación
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Calcular la ganancia de Kalman
        self.x = self.x + np.dot(K, y)  # Actualizar el estado con la innovación ponderada
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # Actualizar la covarianza del estado

# Parámetros del modelo del Filtro de Kalman
dt = 1.0  # Intervalo de tiempo entre mediciones
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado (considera posición y velocidad)
H = np.array([[1, 0]])  # Matriz de observación (solo mide posición)
Q = np.array([[0.1, 0], [0, 0.1]])  # Covarianza del proceso (incertidumbre en la dinámica del sistema)
R = np.array([[0.1]])  # Covarianza de la medición (incertidumbre en las observaciones)
x0 = np.array([[0], [0]])  # Estado inicial (posición y velocidad iniciales)
P0 = np.array([[1, 0], [0, 1]])  # Covarianza inicial del estado

# Crear una instancia del Filtro de Kalman
kf = KalmanFilter(A, H, Q, R, x0, P0)

# Generar datos de ejemplo para el seguimiento de posición
np.random.seed(0)  # Fijar la semilla para reproducibilidad
num_steps = 50  # Número de pasos de tiempo
true_position = 0.1 * np.arange(num_steps)  # Posición verdadera en cada paso
measurements = true_position + np.random.normal(0, 0.5, num_steps)  # Mediciones simuladas con ruido

# Ejecutar el filtro de Kalman en las mediciones
estimated_positions = []
for z in measurements:
    kf.predict()  # Paso de predicción
    kf.update(z)  # Paso de actualización con la nueva medición
    estimated_positions.append(kf.x[0, 0])  # Guardar la posición estimada

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(true_position, label='Posición Verdadera')  # Graficar la posición verdadera
plt.plot(measurements, 'ro', label='Mediciones')  # Graficar las mediciones con ruido
plt.plot(estimated_positions, label='Posición Estimada')  # Graficar la posición estimada por el filtro de Kalman
plt.title('Filtro de Kalman: Predicción')  # Título de la gráfica
plt.xlabel('Paso de Tiempo')  # Etiqueta para el eje x
plt.ylabel('Posición')  # Etiqueta para el eje y
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar la gráfica
