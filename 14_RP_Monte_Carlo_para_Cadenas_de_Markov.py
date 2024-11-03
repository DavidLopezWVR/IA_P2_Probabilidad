#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar

# Definir la función de densidad de probabilidad objetivo (normal estándar)
# Esta función representa la distribución de probabilidad que queremos muestrear
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # Densidad de la normal estándar en x

# Definir la función de densidad de probabilidad de propuesta (normal con desviación sigma)
# Esta función representa una distribución normal centrada en 0 con desviación estándar sigma
def proposal_distribution(x, sigma):
    return np.exp(-0.5 * (x / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)  # Densidad de la normal con desviación sigma

# Implementar el muestreador de Metropolis-Hastings
# Este algoritmo genera muestras de la distribución objetivo utilizando una distribución de propuesta
def metropolis_hastings(target_distribution, proposal_distribution, num_samples, sigma):
    samples = []  # Lista para almacenar las muestras generadas
    current_sample = np.random.normal(0, 1)  # Inicializar con una muestra aleatoria de la normal estándar
    for _ in range(num_samples):  # Generar num_samples muestras
        # Generar una muestra propuesta de la distribución de propuesta centrada en la muestra actual
        proposed_sample = np.random.normal(current_sample, sigma)
        # Calcular la razón de aceptación (aceptance_ratio)
        acceptance_ratio = (target_distribution(proposed_sample) * proposal_distribution(current_sample, sigma)) / \
                           (target_distribution(current_sample) * proposal_distribution(proposed_sample, sigma))
        # Aceptar o rechazar la muestra propuesta
        if np.random.uniform(0, 1) < acceptance_ratio:  # Si el número aleatorio es menor que acceptance_ratio, aceptamos
            current_sample = proposed_sample  # La muestra actual se convierte en la muestra propuesta
        samples.append(current_sample)  # Almacenar la muestra actual (aceptada o rechazada)
    return samples

# Generar muestras utilizando el muestreador de Metropolis-Hastings
num_samples = 10000  # Número de muestras a generar
sigma = 0.5  # Parámetro de la distribución de propuesta (controla la "anchura" de los saltos)
samples = metropolis_hastings(target_distribution, proposal_distribution, num_samples, sigma)

# Graficar el histograma de las muestras generadas para observar la distribución obtenida
plt.hist(samples, bins=50, density=True, alpha=0.5, color='g', label='Muestras')
# Graficar la función de densidad de probabilidad objetivo (normal estándar) para comparar
x = np.linspace(-5, 5, 100)
plt.plot(x, target_distribution(x), color='r', label='Distribución Objetivo')
plt.xlabel('Valor')
plt.ylabel('Densidad de Probabilidad')
plt.title('Muestreador de Metropolis-Hastings')
plt.legend()
plt.show()
