#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Importar matplotlib para crear gráficos

# Definir la dinámica del sistema
def transition_model(x_prev, u):
    # Modelo de transición que actualiza el estado anterior x_prev
    # Sumando la entrada u y un ruido gaussiano con media 0 y desviación estándar 0.1
    return x_prev + u + np.random.normal(0, 0.1, size=x_prev.shape)

# Definir el modelo de observación
def observation_model(x):
    # Modelo de observación que añade ruido gaussiano a la salida del estado x
    return x + np.random.normal(0, 0.1, size=x.shape)

# Generar datos de ejemplo
np.random.seed(0)  # Establecer la semilla para la reproducibilidad
true_states = []  # Lista para almacenar los estados verdaderos
observations = []  # Lista para almacenar las observaciones
num_steps = 50  # Número de pasos de tiempo
initial_state = np.array([0])  # Estado inicial del sistema
for _ in range(num_steps):  # Bucle para simular el sistema
    true_state = transition_model(initial_state, np.array([0.1]))  # Generar el siguiente estado verdadero
    observation = observation_model(true_state)  # Generar la observación correspondiente
    true_states.append(true_state)  # Almacenar el estado verdadero
    observations.append(observation)  # Almacenar la observación
    initial_state = true_state  # Actualizar el estado inicial para la próxima iteración

# Inicializar partículas
num_particles = 100  # Número de partículas en el filtro
particles = np.random.normal(0, 1, size=(num_particles, 1))  # Generar partículas iniciales con distribución normal

# Implementar el filtro de partículas
for t in range(num_steps):  # Para cada paso de tiempo
    # Predicción
    particles = transition_model(particles, np.random.normal(0, 0.1, size=(num_particles, 1)))  # Actualizar partículas

    # Actualización de pesos
    # Calcular los pesos de las partículas en función de la distancia entre las observaciones y las partículas
    weights = np.exp(-0.5 * np.sum((observations[t] - particles)**2, axis=1))  
    weights /= np.sum(weights)  # Normalizar los pesos para que sumen 1

    # Remuestreo
    # Elegir partículas según sus pesos calculados
    indices = np.random.choice(range(num_particles), size=num_particles, p=weights)  # Selección aleatoria
    particles = particles[indices]  # Mantener solo las partículas seleccionadas

# Estimación del estado
estimated_state = np.mean(particles)  # Calcular el estado estimado como la media de las partículas

# Graficar los resultados
plt.figure(figsize=(10, 6))  # Configurar el tamaño de la figura
plt.plot(range(num_steps), [x[0] for x in true_states], label='Estado Verdadero', color='blue')  # Graficar el estado verdadero
plt.scatter(range(num_steps), observations, label='Observaciones', color='red', marker='x')  # Graficar las observaciones
plt.axhline(y=estimated_state, linestyle='--', color='green', label='Estado Estimado')  # Línea horizontal para el estado estimado
plt.xlabel('Paso de Tiempo')  # Etiqueta del eje X
plt.ylabel('Valor')  # Etiqueta del eje Y
plt.title('Filtrado de Partículas')  # Título del gráfico
plt.legend()  # Mostrar la leyenda del gráfico
plt.grid(True)  # Mostrar la cuadrícula en el gráfico
plt.show()  # Mostrar el gráfico
