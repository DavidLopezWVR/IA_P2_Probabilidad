#21110344  David López Rojas  6E2

import numpy as np  # Importar NumPy para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Importar Matplotlib para visualización

# Función para generar partículas aleatorias
def generate_particles(num_particles, x_range, y_range):
    particles = []  # Inicializar una lista para almacenar las partículas
    for _ in range(num_particles):
        # Generar coordenadas x e y aleatorias dentro de los rangos especificados
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        particles.append([x, y, 1.0/num_particles])  # Cada partícula tiene [x, y, peso]
    return np.array(particles)  # Devolver las partículas como un array de NumPy

# Función para mover las partículas
def move_particles(particles, delta_x, delta_y):
    for i in range(len(particles)):
        # Mover cada partícula en x e y, agregando ruido normal aleatorio
        particles[i][0] += np.random.normal(delta_x, 1)  # Mover en x
        particles[i][1] += np.random.normal(delta_y, 1)  # Mover en y
    return particles  # Devolver las partículas movidas

# Función para calcular la probabilidad de observación dada la posición
def observation_prob(observation, particle):
    # En este ejemplo, simplemente se asume una probabilidad constante
    return 1.0  # Retornar una probabilidad fija (se puede modificar según el caso)

# Función para actualizar los pesos de las partículas según las observaciones
def update_weights(particles, observation):
    for i in range(len(particles)):
        # Calcular la probabilidad de observación para cada partícula
        prob = observation_prob(observation, particles[i])
        particles[i][2] *= prob  # Actualizar el peso de la partícula
    total_weight = sum(particle[2] for particle in particles)  # Sumar todos los pesos
    for i in range(len(particles)):
        # Normalizar los pesos para que sumen 1
        particles[i][2] /= total_weight  
    return particles  # Devolver partículas con pesos actualizados

# Función para resamplear las partículas
def resample_particles(particles):
    num_particles = len(particles)  # Obtener el número de partículas
    new_particles = []  # Inicializar lista para las nuevas partículas
    cumulative_weights = np.cumsum([particle[2] for particle in particles])  # Calcular pesos acumulativos
    for _ in range(num_particles):
        rand_val = np.random.uniform(0, 1)  # Generar un valor aleatorio entre 0 y 1
        idx = np.searchsorted(cumulative_weights, rand_val)  # Encontrar el índice correspondiente al valor aleatorio
        new_particles.append(particles[idx].copy())  # Agregar una copia de la partícula seleccionada
    return np.array(new_particles)  # Devolver las nuevas partículas como un array de NumPy

# Función para estimar la posición del robot
def estimate_position(particles):
    x_est = np.mean([particle[0] for particle in particles])  # Calcular la media de las coordenadas x
    y_est = np.mean([particle[1] for particle in particles])  # Calcular la media de las coordenadas y
    return x_est, y_est  # Devolver la estimación de posición

# Parámetros de la simulación
num_particles = 1000  # Número total de partículas a generar
x_range = (0, 100)  # Rango de coordenadas x
y_range = (0, 100)  # Rango de coordenadas y

# Generar partículas iniciales
particles = generate_particles(num_particles, x_range, y_range)  # Llamar a la función para generar partículas

# Movimiento del robot (delta_x, delta_y)
delta_x = 1  # Desplazamiento en x
delta_y = 1  # Desplazamiento en y

# Observaciones del entorno
observation = (50, 50)  # En este ejemplo, el robot observa un objeto en (50, 50)

# Actualizar pesos de las partículas según la observación
particles = update_weights(particles, observation)  # Actualizar los pesos de las partículas

# Resamplear las partículas
particles = resample_particles(particles)  # Resamplear las partículas para obtener un nuevo conjunto

# Estimar la posición del robot
x_est, y_est = estimate_position(particles)  # Estimar la posición del robot
print("Estimación de la posición del robot:", x_est, y_est)  # Imprimir la estimación

# Visualización de las partículas y la estimación de la posición del robot
plt.scatter(particles[:, 0], particles[:, 1], s=5, color='blue', alpha=0.5, label='Partículas')  # Dibujar partículas
plt.scatter(x_est, y_est, color='red', marker='x', label='Estimación')  # Dibujar la estimación de la posición
plt.xlabel('X')  # Etiqueta del eje x
plt.ylabel('Y')  # Etiqueta del eje y
plt.title('Localización de Monte Carlo')  # Título del gráfico
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar la cuadrícula
plt.show()  # Mostrar el gráfico
