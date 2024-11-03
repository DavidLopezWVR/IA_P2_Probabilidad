#21110344  David López Rojas  6E2

import numpy as np

# Función para calcular la probabilidad condicionada
def conditional_probability(data, condition, event):
    # Filtrar los datos que cumplen con la condición especificada
    filtered_data = data[data[:, 0] == condition]
    
    # Contar el número de veces que ocurre el evento dado la condición
    event_count = np.sum(filtered_data[:, 1] == event)
    
    # Contar el número total de veces que ocurre la condición
    condition_count = len(filtered_data)
    
    # Calcular la probabilidad condicionada
    if condition_count > 0:
        conditional_prob = event_count / condition_count  # Proporción del evento dado la condición
    else:
        conditional_prob = 0.0  # Si no hay ocurrencias de la condición, probabilidad es 0
    
    return conditional_prob

# Función para normalizar los datos
def normalize_data(data):
    # Sumar las frecuencias de los eventos (segunda columna)
    total_count = np.sum(data[:, 1])
    
    # Normalizar los datos dividiendo cada frecuencia por la suma total
    normalized_data = data.copy()  # Hacer una copia para evitar modificar los datos originales
    normalized_data[:, 1] /= total_count  # Normaliza la segunda columna (frecuencias)
    
    return normalized_data

# Datos de ejemplo: horas de estudio y resultado del examen (1 = aprobado, 0 = reprobado)
data = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]])

# Calcular la probabilidad condicionada de aprobar dado que se estudiaron 5 horas
hours_studied = 5
probability_pass_given_5_hours = conditional_probability(data, hours_studied, 1)
print("Probabilidad de aprobar dado que se estudiaron 5 horas:", probability_pass_given_5_hours)

# Normalizar los datos
normalized_data = normalize_data(data)
print("\nDatos normalizados:")
print(normalized_data)
