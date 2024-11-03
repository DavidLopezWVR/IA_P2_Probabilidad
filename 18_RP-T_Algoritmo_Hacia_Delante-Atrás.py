#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para realizar operaciones matemáticas y trabajar con matrices

# Definir el modelo oculto de Markov (HMM)

# Matriz de transición de estados, define las probabilidades de cambiar de un estado a otro
transition_matrix = np.array([[0.7, 0.3],  # Probabilidades de transición desde el estado 0 (estado A)
                              [0.4, 0.6]]) # Probabilidades de transición desde el estado 1 (estado B)

# Matriz de emisión, define las probabilidades de observar cada posible observación desde cada estado
emission_matrix = np.array([[0.9, 0.1],  # Probabilidades de emitir cada observación desde el estado A
                             [0.2, 0.8]]) # Probabilidades de emitir cada observación desde el estado B

# Probabilidades iniciales de cada estado al inicio del proceso
initial_state_probabilities = np.array([0.5, 0.5])  # Probabilidad inicial de estar en cada estado (A o B)

# Observaciones: secuencia de eventos observados. Aquí, 0 y 1 representan diferentes observaciones.
observations = [0, 1, 0, 1]  # Ejemplo de secuencia de observaciones, donde 0 = cabeza y 1 = cola

# Implementación del algoritmo hacia adelante
def forward_algorithm(obs, initial_probs, trans_probs, emit_probs):
    num_states = len(initial_probs)  # Número de estados posibles
    num_obs = len(obs)  # Número de observaciones en la secuencia
    
    # Inicializar matriz hacia adelante (almacenará probabilidades de estado en cada tiempo t)
    forward = np.zeros((num_states, num_obs))
    
    # Paso hacia adelante: calcular probabilidades iniciales en el primer tiempo
    forward[:, 0] = initial_probs * emit_probs[:, obs[0]]
    
    # Iterar sobre cada tiempo t para calcular probabilidades hacia adelante en el resto de la secuencia
    for t in range(1, num_obs):
        for j in range(num_states):
            # Actualizar la probabilidad hacia adelante en el tiempo t para el estado j
            forward[j, t] = np.sum(forward[:, t-1] * trans_probs[:, j]) * emit_probs[j, obs[t]]
    
    return forward  # Devolver la matriz hacia adelante

# Implementación del algoritmo hacia atrás
def backward_algorithm(obs, trans_probs, emit_probs):
    num_states = trans_probs.shape[0]  # Número de estados posibles
    num_obs = len(obs)  # Número de observaciones en la secuencia
    
    # Inicializar matriz hacia atrás (almacenará probabilidades en cada tiempo t)
    backward = np.zeros((num_states, num_obs))
    backward[:, -1] = 1  # Iniciar con probabilidad 1 en el último tiempo (base del paso hacia atrás)
    
    # Paso hacia atrás: calcular probabilidades desde el final de la secuencia hasta el inicio
    for t in range(num_obs - 2, -1, -1):  # Iterar de derecha a izquierda en la secuencia de observaciones
        for i in range(num_states):
            # Actualizar la probabilidad hacia atrás en el tiempo t para el estado i
            backward[i, t] = np.sum(trans_probs[i, :] * emit_probs[:, obs[t+1]] * backward[:, t+1])
    
    return backward  # Devolver la matriz hacia atrás

# Calcular la probabilidad total de observar la secuencia dada usando los algoritmos hacia adelante y hacia atrás
forward_probs = forward_algorithm(observations, initial_state_probabilities, transition_matrix, emission_matrix)
backward_probs = backward_algorithm(observations, transition_matrix, emission_matrix)

# Calcular la probabilidad total usando la suma sobre las probabilidades hacia adelante y hacia atrás
total_probability = np.sum(forward_probs[:, -1] * initial_state_probabilities * emission_matrix[:, observations[0]] * backward_probs[:, 0])
print("La probabilidad total de la secuencia de observaciones es:", total_probability)
