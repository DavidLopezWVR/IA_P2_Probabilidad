#21110344  David López Rojas  6E2

from sklearn.datasets import load_iris  # Importar la función para cargar el conjunto de datos Iris
from sklearn.model_selection import train_test_split  # Importar la función para dividir los datos en entrenamiento y prueba
from sklearn.neighbors import KNeighborsClassifier  # Importar el clasificador KNN
from sklearn.metrics import accuracy_score  # Importar la función para calcular la precisión

# Cargar el conjunto de datos Iris de ejemplo
iris = load_iris()  # Cargar el conjunto de datos Iris
X, y = iris.data, iris.target  # Separar las características (X) y las etiquetas de clase (y)

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Dividir los datos: 80% para entrenamiento y 20% para prueba. El parámetro random_state asegura que la división sea reproducible.

# Inicializar y entrenar el clasificador de vecinos más cercanos (KNN)
k = 3  # Número de vecinos a considerar
knn_classifier = KNeighborsClassifier(n_neighbors=k)  # Crear una instancia del clasificador KNN
knn_classifier.fit(X_train, y_train)  # Ajustar el clasificador a los datos de entrenamiento

# Realizar predicciones en los datos de prueba
y_pred = knn_classifier.predict(X_test)  # Predecir las etiquetas de clase para los datos de prueba

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)  # Calcular la precisión comparando las predicciones con las etiquetas reales
print("Precisión del modelo KNN:", accuracy)  # Imprimir la precisión del modelo
