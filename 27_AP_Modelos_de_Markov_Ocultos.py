#21110344  David López Rojas  6E2

from hmmlearn import hmm  # Importar el módulo hmm de la biblioteca hmmlearn para trabajar con modelos ocultos de Markov
import numpy as np  # Importar la biblioteca NumPy para realizar operaciones numéricas

# Definir el modelo HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full")  
# Crear un modelo de HMM con 2 componentes (estados ocultos) y tipo de covarianza "full"

# Parámetros del modelo
model.startprob_ = np.array([0.5, 0.5])  # Establecer las probabilidades iniciales de cada estado (50% para cada uno)
model.transmat_ = np.array([[0.7, 0.3],  # Definir la matriz de transición de estado
                            [0.4, 0.6]])  # Las probabilidades de transición entre los estados
# La primera fila indica que desde el estado 0 hay un 70% de probabilidad de permanecer en el estado 0 y un 30% de pasar al estado 1
# La segunda fila indica que desde el estado 1 hay un 40% de probabilidad de pasar al estado 0 y un 60% de permanecer en el estado 1

model.means_ = np.array([[0.0, 0.0],    # Definir las medias de las distribuciones Gaussianas de cada estado
                         [1.0, 1.0]])  # Estado 0 tiene media (0,0) y el estado 1 tiene media (1,1)

model.covars_ = np.tile(np.identity(2), (2, 1, 1))  
# Establecer la matriz de covarianza para las distribuciones Gaussianas
# Aquí, se usa una matriz de identidad, lo que implica que las características son independientes
# 'np.tile' repite la matriz de identidad 2 veces para cada estado

# Generar secuencia de observaciones
X, Z = model.sample(100)  
# Generar 100 observaciones a partir del modelo HMM
# X contiene las observaciones generadas y Z contiene la secuencia de estados ocultos

# Ajustar el modelo HMM a los datos de ejemplo
model.fit(X)  
# Entrenar el modelo en las observaciones generadas para estimar sus parámetros

# Predicción del estado más probable
new_X = np.array([[0.1, 0.2], [0.8, 0.9]])  
# Definir nuevas observaciones para las cuales se desea predecir el estado más probable
predicted_states = model.predict(new_X)  
# Predecir el estado más probable para las nuevas observaciones usando el modelo ajustado

print("Estado más probable para nuevas observaciones:", predicted_states)  
# Imprimir los estados más probables correspondientes a las nuevas observaciones
