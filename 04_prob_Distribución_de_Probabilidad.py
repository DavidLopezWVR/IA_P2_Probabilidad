#21110344  David López Rojas  6E2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros de la distribución normal
mean = 0  # Media de la distribución normal
std_dev = 1  # Desviación estándar de la distribución normal

# Generar muestras aleatorias de una distribución normal
samples = np.random.normal(mean, std_dev, 1000)  # Genera 1000 muestras

# Calcular la función de densidad de probabilidad (PDF) teórica
x = np.linspace(-5, 5, 100)  # Crear un rango de valores de -5 a 5
pdf = norm.pdf(x, mean, std_dev)  # Calcula la PDF teórica para esos valores

# Trama del histograma de las muestras aleatorias
plt.hist(samples, bins=30, density=True, alpha=0.5, color='g', label='Histograma de Muestras')
# `bins=30` especifica el número de intervalos en el histograma
# `density=True` normaliza el histograma para que el área total sea 1
# `alpha=0.5` establece la transparencia del histograma

# Trama de la función de densidad de probabilidad (PDF) teórica
plt.plot(x, pdf, color='r', label='PDF Teórica')  # Traza la PDF en rojo

# Etiquetas y leyenda
plt.xlabel('Valor')  # Etiqueta para el eje x
plt.ylabel('Densidad de Probabilidad')  # Etiqueta para el eje y
plt.title('Distribución Normal')  # Título de la gráfica
plt.legend()  # Muestra la leyenda en la gráfica

# Mostrar la gráfica
plt.show()  # Renderiza y muestra la gráfica
