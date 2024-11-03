#21110344  David López Rojas  6E2

from hmmlearn import hmm  # Importar la biblioteca hmmlearn para trabajar con modelos ocultos de Markov
import numpy as np  # Importar la biblioteca NumPy para realizar operaciones matemáticas y trabajar con matrices

# Definir el modelo HMM con distribuciones Gaussianas en cada estado
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Configurar los parámetros del modelo HMM

# Definir las probabilidades iniciales de estar en cada estado al inicio del proceso
model.startprob_ = np.array([0.5, 0.5])  # Asumimos que ambos estados tienen una probabilidad inicial del 50%

# Definir la matriz de transición, que indica la probabilidad de moverse de un estado a otro
model.transmat_ = np.array([[0.7, 0.3],  # Probabilidades de transición desde el estado 0
                            [0.4, 0.6]]) # Probabilidades de transición desde el estado 1

# Definir las medias de las distribuciones Gaussianas para cada estado
model.means_ = np.array([[0.0, 0.0],    # Media de la distribución Gaussiana para el estado 0
                         [1.0, 1.0]])   # Media de la distribución Gaussiana para el estado 1

# Definir las covarianzas de las distribuciones Gaussianas en cada estado
model.covars_ = np.tile(np.identity(2), (2, 1, 1))  # Covarianza identidad en ambos estados, matriz 2x2 para cada estado

# Generar datos de ejemplo simulando una secuencia de observaciones y sus estados ocultos
X, Z = model.sample(100)  # 100 muestras generadas usando los parámetros iniciales del modelo

# Ajustar el modelo a los datos generados (entrenamiento del HMM)
model.fit(X)

# Predecir el estado más probable para nuevas observaciones basadas en el modelo ajustado
new_X = np.array([[0.1, 0.2], [0.8, 0.9]])  # Nuevas observaciones para las cuales se quieren predecir los estados
predicted_states = model.predict(new_X)  # Obtener los estados más probables para las nuevas observaciones

# Imprimir los estados más probables para las nuevas observaciones
print("Estado más probable para nuevas observaciones:", predicted_states)
