#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para realizar operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar
from sklearn.datasets import make_blobs  # Importar la función para generar datos sintéticos
from sklearn.cluster import KMeans  # Importar el algoritmo K-Means para agrupamiento

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# 'n_samples=300' indica que se generarán 300 muestras
# 'centers=4' especifica que habrá 4 centros (agrupaciones) en los datos
# 'cluster_std=0.60' define la desviación estándar de las agrupaciones, afectando su dispersión
# 'random_state=0' asegura que la generación de datos sea reproducible

# Inicializar y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=4)  # Crear una instancia del modelo K-Means con 4 grupos
kmeans.fit(X)  # Ajustar el modelo a los datos generados

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_  # Obtener las etiquetas asignadas a cada punto de datos
centers = kmeans.cluster_centers_  # Obtener las coordenadas de los centroides de los clusters

# Graficar los clusters y los centroides
plt.figure(figsize=(8, 6))  # Crear una nueva figura con un tamaño específico
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)  # Graficar los puntos de datos
# 'c=labels' colorea los puntos según sus etiquetas de cluster
# 'cmap='viridis'' especifica el mapa de colores a usar
# 's=50' define el tamaño de los puntos
# 'alpha=0.7' establece la transparencia de los puntos

plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=200, edgecolors='k')  # Graficar los centroides
# 'marker='o'' establece el marcador como un círculo
# 'c='red'' colorea los centroides de rojo
# 's=200' define un tamaño mayor para los centroides
# 'edgecolors='k'' establece el borde de los marcadores de los centroides como negro

plt.title('Agrupamiento con K-Means')  # Añadir un título al gráfico
plt.xlabel('Feature 1')  # Etiqueta del eje x
plt.ylabel('Feature 2')  # Etiqueta del eje y
plt.show()  # Mostrar el gráfico generado
