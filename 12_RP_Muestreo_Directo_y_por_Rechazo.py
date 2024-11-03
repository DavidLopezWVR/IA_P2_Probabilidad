#21110344  David López Rojas  6E2

from scipy.stats import norm  # Importar la función para la distribución normal

# Función de densidad de probabilidad de la distribución objetivo (normal con media `mu` y desviación estándar `sigma`)
def target_distribution(x):
    return norm.pdf(x, mu, sigma)  # Calcula la densidad de la normal estándar en x

# Función de densidad de probabilidad de la distribución de propuesta (uniforme en el intervalo [-3, 3])
def proposal_distribution(x):
    # Retorna la densidad de una distribución uniforme en [-3, 3] (1/6 dentro del intervalo y 0 fuera)
    return 1 / 6 if -3 <= x <= 3 else 0

# Realizar muestreo por rechazo
samples_rejection = []  # Lista para almacenar las muestras aceptadas
for _ in range(1000):  # Generar 1000 muestras
    # Generar una muestra de la distribución de propuesta (uniforme en [-3, 3])
    sample = np.random.uniform(-3, 3)
    
    # Calcular la razón entre la función de densidad de la distribución objetivo y la distribución de propuesta
    ratio = target_distribution(sample) / proposal_distribution(sample)
    
    # Aceptar o rechazar la muestra con probabilidad proporcional a la proporción calculada
    # Si un número aleatorio entre 0 y 1 es menor que `ratio`, aceptamos la muestra
    if np.random.uniform(0, 1) < ratio:
        samples_rejection.append(sample)  # Agregar la muestra aceptada a la lista

# Imprimir las primeras 10 muestras aceptadas
print("\nMuestras generadas por muestreo por rechazo:")
print(samples_rejection[:10])
