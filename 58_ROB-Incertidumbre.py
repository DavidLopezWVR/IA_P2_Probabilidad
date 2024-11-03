#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importar matplotlib para visualización

# Parámetros del filtro de Kalman
dt = 0.1  # Paso de tiempo
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado (modelo de movimiento)
B = np.array([[0.5*dt**2], [dt]])  # Matriz de control (no se utiliza control en este ejemplo)
H = np.array([[1, 0]])  # Matriz de observación (observamos solo la posición)
Q = np.array([[0.1, 0], [0, 0.1]])  # Covarianza del proceso (ruido del sistema)
R = np.array([[1]])  # Covarianza de la medición (ruido del sensor)

# Estado inicial y covarianza inicial
x = np.array([[0], [0]])  # Estado inicial [posición, velocidad]
P = np.eye(2) * 10  # Covarianza inicial (alta incertidumbre)

# Simulación de la trayectoria verdadera y observaciones
true_position = []  # Lista para almacenar la posición verdadera
measurements = []  # Lista para almacenar las observaciones
for i in range(100):
    true_position.append(x[0, 0])  # Almacenar la posición verdadera
    z = H @ x + np.random.normal(0, np.sqrt(R[0, 0]))  # Generar observación con ruido
    measurements.append(z[0, 0])  # Almacenar la medición
    u = np.array([[0]])  # No hay control (aceleración cero)
    x = A @ x + B @ u + np.random.multivariate_normal([0, 0], Q).reshape((2, 1))
    # Actualizar el estado verdadero con el modelo de movimiento y ruido del proceso

# Filtro de Kalman
filtered_position = []  # Lista para almacenar la posición estimada
for z in measurements:
    # Predicción del estado y covarianza
    x_pred = A @ x  # Predicción del siguiente estado
    P_pred = A @ P @ A.T + Q  # Predicción de la covarianza
    
    # Actualización utilizando la observación
    y = z - H @ x_pred  # Residuo de la medición
    S = H @ P_pred @ H.T + R  # Covarianza del residuo
    K = P_pred @ H.T @ np.linalg.inv(S)  # Ganancia de Kalman
    x = x_pred + K @ y  # Actualizar el estado con la ganancia de Kalman
    P = (np.eye(2) - K @ H) @ P_pred  # Actualizar la covarianza
    
    filtered_position.append(x[0, 0])  # Almacenar la posición filtrada

# Visualización de la trayectoria verdadera y la estimada por el filtro de Kalman
plt.plot(true_position, label='Trayectoria Verdadera')  # Traza la trayectoria verdadera
plt.plot(measurements, 'ro', label='Observaciones')  # Traza las observaciones (ruidosas)
plt.plot(filtered_position, label='Estimación del Filtro de Kalman')  # Traza la estimación del filtro
plt.xlabel('Tiempo')  # Etiqueta del eje X
plt.ylabel('Posición')  # Etiqueta del eje Y
plt.title('Filtro de Kalman para Estimación de Posición')  # Título del gráfico
plt.legend()  # Muestra la leyenda
plt.grid(True)  # Muestra la cuadrícula
plt.show()  # Muestra el gráfico
