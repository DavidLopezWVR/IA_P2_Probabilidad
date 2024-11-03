#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para crear gráficos
from sklearn.datasets import make_classification  # Importar la función para generar datos de clasificación
from sklearn.svm import SVC  # Importar la clase SVC (Support Vector Classifier) para clasificación con SVM

# Generar datos de ejemplo con dos clases y características
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)
# n_samples: número total de muestras a generar
# n_features: número de características (en este caso, 2)
# n_classes: número de clases (en este caso, 2)
# n_clusters_per_class: número de clústeres por clase
# random_state: semilla para la reproducibilidad

# Visualizar los datos de ejemplo
plt.figure(figsize=(8, 6))  # Crear una figura con un tamaño específico
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)  # Graficar los puntos
# X[:, 0]: primera característica (eje x)
# X[:, 1]: segunda característica (eje y)
# c=y: color de los puntos basado en las etiquetas de clase
# cmap='viridis': mapa de colores a utilizar
# s=50: tamaño de los puntos
# alpha=0.7: transparencia de los puntos
plt.title('Datos de ejemplo')  # Título del gráfico
plt.xlabel('Feature 1')  # Etiqueta del eje x
plt.ylabel('Feature 2')  # Etiqueta del eje y
plt.show()  # Mostrar el gráfico

# Verificar la separabilidad lineal utilizando un clasificador SVM lineal
svm_classifier = SVC(kernel='linear')  # Crear un clasificador SVM con un núcleo lineal
svm_classifier.fit(X, y)  # Ajustar el clasificador a los datos generados

# Visualizar el hiperplano resultante
w = svm_classifier.coef_[0]  # Obtener los coeficientes (pesos) del modelo
a = -w[0] / w[1]  # Calcular la pendiente del hiperplano
xx = np.linspace(-2, 2)  # Crear un rango de valores para el eje x
yy = a * xx - (svm_classifier.intercept_[0]) / w[1]  # Calcular los valores correspondientes en el eje y para el hiperplano

plt.figure(figsize=(8, 6))  # Crear una nueva figura para visualizar el hiperplano
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)  # Graficar nuevamente los datos
plt.plot(xx, yy, 'k-')  # Graficar el hiperplano en negro ('k-')
plt.title('Separabilidad Lineal')  # Título del gráfico
plt.xlabel('Feature 1')  # Etiqueta del eje x
plt.ylabel('Feature 2')  # Etiqueta del eje y
plt.show()  # Mostrar el gráfico
