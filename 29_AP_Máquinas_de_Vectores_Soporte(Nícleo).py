#21110344  David López Rojas  6e2

import numpy as np  # Importar la biblioteca NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Importar matplotlib para visualización de datos
from sklearn.datasets import make_circles  # Importar la función para generar datos en forma de círculos
from sklearn.svm import SVC  # Importar el clasificador SVM

# Generar datos de ejemplo (círculos)
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)  
# Crear un conjunto de datos en forma de círculos con 100 muestras, un poco de ruido y un factor que determina la separación entre los círculos

# Crear un clasificador SVM con un núcleo RBF (radial basis function)
svm_classifier = SVC(kernel='rbf', C=10, gamma='auto')  
# Inicializar el clasificador SVM utilizando un núcleo RBF, C como un parámetro de penalización y gamma configurado automáticamente

# Ajustar el clasificador a los datos de ejemplo
svm_classifier.fit(X, y)  # Entrenar el clasificador SVM con los datos generados

# Función para visualizar la frontera de decisión
def plot_decision_boundary(clf, X, y):  
    plt.figure(figsize=(8, 6))  # Crear una nueva figura para la visualización
    h = .02  # Definir el tamaño del paso en la malla
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1  # Definir los límites del eje x con un pequeño margen
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1  # Definir los límites del eje y con un pequeño margen
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  
    # Crear una malla de puntos para evaluar la frontera de decisión del clasificador

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Predecir las clases para cada punto en la malla
    Z = Z.reshape(xx.shape)  # Remodelar la salida para que coincida con la forma de la malla

    plt.contourf(xx, yy, Z, alpha=0.3)  # Graficar la frontera de decisión como un contorno relleno con transparencia
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k')  
    # Graficar los puntos de datos originales coloreados según sus etiquetas
    plt.xlabel('Feature 1')  # Etiqueta del eje x
    plt.ylabel('Feature 2')  # Etiqueta del eje y
    plt.title('Support Vector Machine with RBF Kernel')  # Título del gráfico
    plt.show()  # Mostrar la figura generada

# Visualizar la frontera de decisión
plot_decision_boundary(svm_classifier, X, y)  # Llamar a la función para graficar la frontera de decisión del clasificador SVM
