#21110344  David López Rojas  6E2

import numpy as np  # Importar NumPy para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Importar Matplotlib para visualización de gráficos

# Función para actualizar el estado y la covarianza del filtro de Kalman extendido (EKF)
def ekf_update(mu, sigma, z, R, H):
    # Paso de predicción
    mu_bar = mu  # El estado previsto es el mismo que el estado actual
    sigma_bar = sigma  # La covarianza prevista es la misma que la covarianza actual
    
    # Paso de actualización
    K = sigma_bar @ H.T @ np.linalg.inv(H @ sigma_bar @ H.T + R)  # Calcular la ganancia de Kalman
    mu = mu_bar + K @ (z - H @ mu_bar)  # Actualizar el estado con la medición
    sigma = (np.eye(len(mu)) - K @ H) @ sigma_bar  # Actualizar la covarianza

    return mu, sigma  # Devolver el estado y la covarianza actualizados

# Parámetros del entorno y del robot
landmarks = np.array([[2, 2], [8, 8], [5, 12]])  # Posiciones conocidas de los hitos
num_landmarks = len(landmarks)  # Número de hitos
robot_start = np.array([0, 0])  # Posición inicial del robot
robot_true_motion = np.array([1, 1])  # Movimiento verdadero del robot
robot_sensor_noise = 0.1  # Ruido del sensor

# Parámetros del filtro de Kalman extendido (EKF)
mu = np.zeros(3 + 2 * num_landmarks)  # Estado inicial [x, y, theta, l1_x, l1_y, l2_x, l2_y, ..., ln_x, ln_y]
sigma = np.eye(3 + 2 * num_landmarks) * 0.1  # Covarianza inicial, pequeña incertidumbre

# Ciclo de tiempo
num_steps = 100  # Número de pasos en la simulación
trajectory = []  # Lista para almacenar la trayectoria del robot

for t in range(num_steps):
    # Movimiento verdadero del robot (en este ejemplo, se supone un movimiento lineal)
    robot_true_motion += np.array([0.1, 0.1])  # Actualizar la posición del robot
    
    # Simular medición de rango (distancia a los hitos) con ruido
    true_distances = np.linalg.norm(landmarks - robot_true_motion, axis=1)  # Calcular distancias verdaderas a los hitos
    noisy_distances = true_distances + np.random.normal(0, robot_sensor_noise, num_landmarks)  # Agregar ruido a las distancias
    
    # Actualizar el estado del filtro de Kalman extendido (EKF)
    mu[0:3], sigma[0:3, 0:3] = ekf_update(mu[0:3], sigma[0:3, 0:3], robot_true_motion, np.eye(3), np.eye(3))
    for i in range(num_landmarks):
        if noisy_distances[i] < 20:  # Solo actualizar si el rango es razonable
            z = np.array([noisy_distances[i]])  # Medición de la distancia con ruido
            landmark_index = 3 + 2 * i  # Índice del hito en el estado
            H = np.zeros((1, 3 + 2 * num_landmarks))  # Inicializar la matriz Jacobiana H
            # Calcular la parte de la matriz Jacobiana correspondiente a la observación del hito
            H[:, 0:3] = -np.array([[(landmarks[i][0] - mu[0]) / true_distances[i], 
                                     (landmarks[i][1] - mu[1]) / true_distances[i], 
                                     0]])
            H[:, landmark_index:landmark_index + 2] = np.array([[(landmarks[i][0] - mu[0]) / true_distances[i], 
                                                                  (landmarks[i][1] - mu[1]) / true_distances[i]]])
            R = np.eye(1) * robot_sensor_noise  # Matriz de covarianza del ruido del sensor
            mu, sigma = ekf_update(mu, sigma, z, R, H)  # Actualizar el estado y la covarianza usando EKF
    
    # Guardar la posición actual del robot
    trajectory.append(mu[0:2])  # Almacenar la posición estimada del robot

# Visualizar el entorno y la trayectoria del robot
plt.figure(figsize=(10, 6))  # Crear una figura para el gráfico
plt.plot(trajectory[0][0], trajectory[0][1], 'go', markersize=10, label='Inicio')  # Marcador de inicio
plt.plot(trajectory[-1][0], trajectory[-1][1], 'ro', markersize=10, label='Fin')  # Marcador de fin
plt.plot(trajectory[:, 0], trajectory[:, 1], '-b', label='Trayectoria del Robot')  # Traza la trayectoria del robot
plt.scatter(landmarks[:, 0], landmarks[:, 1], color='orange', marker='x', label='Hitos')  # Dibujar los hitos
plt.xlabel('X')  # Etiqueta del eje x
plt.ylabel('Y')  # Etiqueta del eje y
plt.title('Simulación SLAM con Filtro de Kalman Extendido')  # Título del gráfico
plt.legend()  # Mostrar leyenda
plt.grid(True)  # Activar la cuadrícula
plt.axis('equal')  # Asegurar proporciones iguales en los ejes
plt.show()  # Mostrar el gráfico
