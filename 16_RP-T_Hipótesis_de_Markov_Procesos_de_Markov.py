#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para manipulación de matrices y operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar

# Definir la matriz de transición para el proceso de Markov
# La matriz de transición indica la probabilidad de moverse de un estado a otro en el siguiente paso
transition_matrix = np.array([[0.8, 0.2],  # Probabilidad de transición de A a A (0.8) y de A a B (0.2)
                              [0.3, 0.7]]) # Probabilidad de transición de B a A (0.3) y de B a B (0.7)

# Definir los nombres de los estados del proceso de Markov
states = ['A', 'B']

# Generar una secuencia de estados para el proceso de Markov
num_steps = 1000  # Número de pasos de tiempo a simular
current_state = np.random.choice(states)  # Seleccionar un estado inicial aleatorio
sequence = [current_state]  # Inicializar la secuencia de estados con el estado inicial

# Realizar la simulación del proceso de Markov
for _ in range(num_steps):  # Repetir el proceso para el número de pasos especificado
    # Elegir el siguiente estado basado en la probabilidad de transición desde el estado actual
    current_state = np.random.choice(states, p=transition_matrix[states.index(current_state)])
    sequence.append(current_state)  # Añadir el estado seleccionado a la secuencia

# Contar la frecuencia de cada estado en la secuencia generada
state_counts = {state: sequence.count(state) for state in states}

# Graficar la secuencia de estados a lo largo del tiempo
plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura
plt.plot(sequence, marker='o', linestyle='-')  # Graficar la secuencia de estados
plt.title('Proceso de Markov: Secuencia de Estados')  # Título de la gráfica
plt.xlabel('Paso de Tiempo')  # Etiqueta para el eje x
plt.ylabel('Estado')  # Etiqueta para el eje y
plt.yticks(range(len(states)), states)  # Etiquetas en el eje y para mostrar los nombres de los estados
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar la gráfica

# Imprimir la frecuencia de cada estado en la secuencia generada
print("Frecuencia de cada estado en la secuencia:")
for state, count in state_counts.items():
    print(f"Estado {state}: {count} veces")  # Imprimir el número de veces que aparece cada estado
