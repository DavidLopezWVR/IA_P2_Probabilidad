#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas
from sklearn.datasets import make_blobs  # Importar la función para generar datos sintéticos en forma de agrupaciones
from sklearn.mixture import GaussianMixture  # Importar la clase para modelos de mezcla gaussiana

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=1.5, random_state=42)
# 'n_samples=1000' indica que se generarán 1000 muestras
# 'centers=3' significa que se crearán 3 agrupaciones (clusters)
# 'cluster_std=1.5' especifica la desviación estándar de las agrupaciones
# 'random_state=42' asegura que la generación de datos sea reproducible

# Inicializar el modelo de mezcla gaussiana
model = GaussianMixture(n_components=3, random_state=42)
# 'n_components=3' indica que el modelo buscará 3 componentes gaussianos en los datos

# Ajustar el modelo a los datos de ejemplo
model.fit(X)  # Entrenar el modelo con los datos generados

# Mostrar los parámetros estimados del modelo
print("Parámetros estimados del modelo:")  # Imprimir un mensaje previo a los parámetros
print("Peso de cada componente:", model.weights_)  # Imprimir los pesos (proporciones) de cada componente
print("Media de cada componente:", model.means_)  # Imprimir las medias de cada componente gaussiano
print("Covarianza de cada componente:", model.covariances_)  # Imprimir las matrices de covarianza de cada componente
