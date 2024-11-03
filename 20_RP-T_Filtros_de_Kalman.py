#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Importar matplotlib para crear gráficos

# Definición de la clase Filtro de Kalman
class KalmanFilter:
    # Inicialización del filtro de Kalman con los parámetros necesarios
    def __init__(self, A, H, Q, R, x0, P0):
        self.A = A  # Matriz de transición de estado (predicción del estado)
        self.H = H  # Matriz de observación (relación entre el estado y las observaciones)
        self.Q = Q  # Covarianza del proceso (incertidumbre en el modelo del proceso)
        self.R = R  # Covarianza de la medición (incertidumbre en las observaciones)
        self.x = x0  # Estado inicial (estimación inicial del estado)
        self.P = P0  # Covarianza inicial (incertidumbre inicial en la estimación)

    # Método para predecir el siguiente estado
    def predict(self):
        self.x = np.dot(self.A, self.x)  # Predicción del nuevo estado
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # Actualización de la covarianza

    # Método para actualizar la estimación con una nueva observación
    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Cálculo del residuo (diferencia entre la observación y la predicción)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Covarianza del residuo
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Ganancia de Kalman
        self.x = self.x + np.dot(K, y)  # Actualización del estado con el residuo ponderado
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # Actualización de la covarianza

# Parámetros del modelo
dt = 1.0  # Intervalo de tiempo entre predicciones
A = np.array([[1, dt], [0, 1]])  # Matriz de transición que describe el modelo de movimiento (posición y velocidad)
H = np.array([[1, 0]])  # Matriz de observación que relaciona el estado con la medida (solo se mide la posición)
Q = np.array([[0.1, 0], [0, 0.1]])  # Covarianza del proceso (modela la incertidumbre en el modelo)
R = np.array([[0.1]])  # Covarianza de la medición (modela la incertidumbre en la medida)
x0 = np.array([[0], [0]])  # Estado inicial (posición y velocidad iniciales)
P0 = np.array([[1, 0], [0, 1]])  # Covarianza inicial (incertidumbre inicial en la estimación del estado)

# Crear una instancia del filtro de Kalman
kf = KalmanFilter(A, H, Q, R, x0, P0)

# Generar datos de ejemplo
np.random.seed(0)  # Establecer una semilla para la reproducibilidad
num_steps = 50  # Número de pasos en el tiempo
true_position = 0.1 * np.arange(num_steps)  # Posición verdadera que sigue una línea recta
measurements = true_position + np.random.normal(0, 0.5, num_steps)  # Medidas observadas con ruido

# Ejecutar el filtro de Kalman para estimar la posición
estimated_positions = []  # Lista para almacenar las posiciones estimadas
for z in measurements:  # Para cada medición
    kf.predict()  # Predicción del siguiente estado
    kf.update(z)  # Actualización del estado con la nueva medición
    estimated_positions.append(kf.x[0, 0])  # Almacenar la estimación de la posición

# Graficar los resultados
plt.figure(figsize=(10, 6))  # Configurar el tamaño de la figura
plt.plot(true_position, label='Posición Verdadera')  # Graficar la posición verdadera
plt.plot(measurements, 'ro', label='Mediciones')  # Graficar las mediciones observadas como puntos rojos
plt.plot(estimated_positions, label='Posición Estimada')  # Graficar la posición estimada
plt.title('Filtro de Kalman: Predicción')  # Título del gráfico
plt.xlabel('Paso de Tiempo')  # Etiqueta del eje X
plt.ylabel('Posición')  # Etiqueta del eje Y
plt.legend()  # Mostrar la leyenda del gráfico
plt.grid(True)  # Mostrar la cuadrícula en el gráfico
plt.show()  # Mostrar el gráfico
