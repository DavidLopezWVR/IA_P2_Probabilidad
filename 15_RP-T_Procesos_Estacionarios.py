#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar

# Parámetros del proceso estacionario
num_samples = 1000  # Número de muestras o puntos en el tiempo
mean = 0  # Media del proceso (centro de los valores generados)
std = 1  # Desviación estándar del proceso (amplitud de las variaciones)

# Generar muestras del proceso estacionario (ruido blanco gaussiano)
# Se genera un conjunto de 1000 muestras siguiendo una distribución normal con media 0 y desviación estándar 1
samples = np.random.normal(mean, std, size=num_samples)

# Graficar las muestras generadas
plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura
plt.plot(samples)  # Graficar los valores de las muestras
plt.title('Proceso Estacionario en Sentido Amplio (Ruido Blanco Gaussiano)')  # Título de la gráfica
plt.xlabel('Tiempo')  # Etiqueta para el eje x, representando el tiempo o el índice de la muestra
plt.ylabel('Valor')  # Etiqueta para el eje y, representando el valor de cada muestra
plt.grid(True)  # Mostrar una cuadrícula para facilitar la lectura
plt.show()  # Mostrar la gráfica
