#21110344  David López Rojas  6E2

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Crear un conjunto de datos de ejemplo
data = np.random.rand(100, 2)  # Generar 100 muestras con 2 características aleatorias entre 0 y 1

# Definir las dimensiones del mapa autoorganizado (SOM)
som_width = 10  # Ancho del mapa SOM
som_height = 10  # Alto del mapa SOM

# Inicializar y entrenar el SOM
som = MiniSom(som_width, som_height, 2, sigma=1.0, learning_rate=0.5)  # Crear un objeto MiniSom
# con dimensiones 10x10 y 2 características de entrada. sigma controla la vecindad y learning_rate la tasa de aprendizaje.
som.random_weights_init(data)  # Inicializar los pesos del SOM de forma aleatoria utilizando los datos de entrada
som.train_random(data, 100)  # Entrenar el SOM con los datos durante 100 iteraciones

# Visualizar el SOM
plt.figure(figsize=(8, 8))  # Crear una figura de tamaño 8x8 pulgadas
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Dibujar el mapa de distancia como fondo, transpuesto
plt.colorbar()  # Añadir una barra de color para representar los valores del mapa de distancia

# Visualizar los datos de entrada y las ubicaciones de los nodos ganadores
for i, x in enumerate(data):
    winner = som.winner(x)  # Obtener la posición del nodo ganador para cada muestra en los datos
    # Dibujar un círculo rojo alrededor de la ubicación del nodo ganador
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='r', markersize=10, markeredgewidth=2)
    
plt.axis([0, som_width, 0, som_height])  # Definir los límites del gráfico
plt.title('Mapa Autoorganizado de Kohonen')  # Título del gráfico
plt.show()  # Mostrar la visualización del SOM
