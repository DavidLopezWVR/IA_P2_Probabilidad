#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para cálculos numéricos
from scipy.stats import norm  # Importar la función para la distribución normal

# Generar datos de ejemplo (100 datos con distribución normal de media 2 y desviación estándar 1)
np.random.seed(0)  # Fijar la semilla para obtener resultados reproducibles
data = np.random.normal(loc=2, scale=1, size=100)

# Definir la función de verosimilitud para una distribución normal
# Calcula la probabilidad conjunta de los datos bajo los parámetros mu y sigma
def likelihood(data, mu, sigma):
    return np.prod(norm.pdf(data, loc=mu, scale=sigma))  # Multiplica las densidades individuales

# Definir la distribución previa para los parámetros (media y desviación estándar)
def prior(mu, sigma):
    # Asumimos una distribución uniforme para la media y la desviación estándar (sin información previa específica)
    return 1  # Devuelve una constante para todos los valores de mu y sigma

# Definir la función de verosimilitud ponderada
# Calcula la verosimilitud ponderada para diferentes combinaciones de mu y sigma
def weighted_likelihood(data, mu_values, sigma_values):
    likelihoods = np.zeros((len(mu_values), len(sigma_values)))  # Inicializar la matriz de verosimilitudes
    for i, mu in enumerate(mu_values):
        for j, sigma in enumerate(sigma_values):
            # Calcular la verosimilitud de los datos para cada par de mu y sigma
            likelihoods[i, j] = likelihood(data, mu, sigma) * prior(mu, sigma)
    # Normalizar las verosimilitudes para que sumen 1 (obteniendo una distribución de probabilidad)
    return likelihoods / np.sum(likelihoods)

# Definir los valores posibles de los parámetros mu y sigma a evaluar
mu_values = np.linspace(0, 4, 100)  # Rango de mu entre 0 y 4
sigma_values = np.linspace(0.1, 2, 100)  # Rango de sigma entre 0.1 y 2

# Calcular la verosimilitud ponderada para cada combinación de mu y sigma
weighted_likelihoods = weighted_likelihood(data, mu_values, sigma_values)

# Encontrar los valores de mu y sigma que maximizan la verosimilitud ponderada
max_likelihood_index = np.unravel_index(np.argmax(weighted_likelihoods), weighted_likelihoods.shape)
estimated_mu = mu_values[max_likelihood_index[0]]  # Mu que maximiza la verosimilitud ponderada
estimated_sigma = sigma_values[max_likelihood_index[1]]  # Sigma que maximiza la verosimilitud ponderada

# Imprimir los resultados de los parámetros estimados
print("Parámetros estimados:")
print("Media estimada:", estimated_mu)
print("Desviación estándar estimada:", estimated_sigma)
