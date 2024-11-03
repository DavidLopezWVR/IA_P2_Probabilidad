#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para realizar operaciones numéricas
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización de datos
from sklearn.datasets import make_blobs  # Importar la función para generar datos sintéticos
from sklearn.neighbors import KNeighborsClassifier  # Importar el clasificador k-NN
from sklearn.cluster import KMeans  # Importar el algoritmo de agrupamiento K-Medias

# Generar datos de ejemplo
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)  
# Crear un conjunto de datos sintéticos con 300 muestras, 4 centros y una desviación estándar de 0.60

# Implementar k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=5)  
# Crear un clasificador k-NN con 5 vecinos más cercanos
knn_classifier.fit(X, y)  # Ajustar el clasificador a los datos generados

# Implementar k-Medias
kmeans = KMeans(n_clusters=4)  # Crear un modelo de K-Medias con 4 clusters
kmeans.fit(X)  # Ajustar el modelo a los datos generados

# Graficar k-NN
plt.figure(figsize=(12, 4))  # Inicializar una figura con un tamaño específico

plt.subplot(1, 2, 1)  # Crear una subgráfica en la primera posición (1 fila, 2 columnas, primer gráfico)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)  
# Graficar los puntos de datos originales coloreados por las etiquetas verdaderas con un tamaño de 50 y transparencia de 0.7
plt.title('k-NN')  # Título del gráfico

# Crear malla para graficar la superficie de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Definir los límites del eje x
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Definir los límites del eje y
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  
# Crear una malla de puntos que cubra el área de los datos con un paso de 0.02

Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])  
# Predecir la clase para cada punto en la malla utilizando el clasificador k-NN
Z = Z.reshape(xx.shape)  # Remodelar la salida para que coincida con la forma de la malla
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  
# Graficar la superficie de decisión como un contorno relleno con transparencia de 0.3

# Graficar k-Medias
plt.subplot(1, 2, 2)  # Crear una subgráfica en la segunda posición (1 fila, 2 columnas, segundo gráfico)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)  
# Graficar los puntos de datos originales coloreados por las etiquetas de los clusters obtenidos de k-Medias
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=200, edgecolors='k')  
# Graficar los centros de los clusters como puntos rojos más grandes y con borde negro
plt.title('k-Medias')  # Título del gráfico

plt.show()  # Mostrar todos los gráficos generados
