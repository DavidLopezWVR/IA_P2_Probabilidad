#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para cálculos numéricos

# Función para la cinemática inversa de un brazo robótico de dos grados de libertad
def inverse_kinematics(x, y, l1, l2):
    # Distancia del efector final al origen
    d = np.sqrt(x**2 + y**2)  # Calcular la distancia desde el origen (0,0) hasta el efector final (x,y)

    # Ángulo entre el primer eslabón y la línea que conecta el origen y el efector final
    alpha = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    # Se utiliza la ley de cosenos para calcular el ángulo alpha

    # Ángulo entre el primer eslabón y la línea x
    beta = np.arctan2(y, x) - np.arctan2(l2 * np.sin(np.pi - alpha), l1 + l2 * np.cos(np.pi - alpha))
    # Calcular el ángulo beta utilizando la función arctan2 para obtener el ángulo correcto en función de las coordenadas

    return beta, np.pi - alpha  # Retornar los ángulos theta1 (beta) y theta2 (np.pi - alpha)

# Posición deseada del efector final
x_desired = 5  # Coordenada x objetivo
y_desired = 5  # Coordenada y objetivo

# Longitudes de los eslabones
l1 = 3  # Longitud del primer eslabón
l2 = 3  # Longitud del segundo eslabón

# Calcular las posiciones de las articulaciones para alcanzar la posición deseada
theta1, theta2 = inverse_kinematics(x_desired, y_desired, l1, l2)  # Llamar a la función de cinemática inversa

# Mostrar los resultados
print("Ángulo de la articulación 1:", np.degrees(theta1))  # Convertir theta1 de radianes a grados y mostrar
print("Ángulo de la articulación 2:", np.degrees(theta2))  # Convertir theta2 de radianes a grados y mostrar
