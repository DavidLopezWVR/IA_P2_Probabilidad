#21110344  David López Rojas  6E2

from sklearn.datasets import load_iris  # Importar la función para cargar el conjunto de datos Iris
from sklearn.model_selection import train_test_split  # Importar la función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.naive_bayes import GaussianNB  # Importar el clasificador Naive Bayes con distribución gaussiana
from sklearn.metrics import accuracy_score  # Importar la función para calcular la precisión del clasificador

# Cargar el conjunto de datos Iris
iris = load_iris()  # Cargar el conjunto de datos Iris en la variable 'iris'
X = iris.data  # Extraer las características (atributos) del conjunto de datos
y = iris.target  # Extraer las etiquetas (clases) del conjunto de datos

# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 'test_size=0.3' indica que el 30% de los datos se utilizarán para prueba
# 'random_state=42' asegura que la división sea reproducible en ejecuciones futuras

# Crear un clasificador Naive Bayes gaussiano
classifier = GaussianNB()  # Inicializar el clasificador

# Entrenar el clasificador con los datos de entrenamiento
classifier.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento

# Realizar predicciones sobre los datos de prueba
predictions = classifier.predict(X_test)  # Predecir las clases de los datos de prueba

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, predictions)  # Comparar las predicciones con las etiquetas reales para calcular la precisión
print("Precisión del clasificador Naive Bayes:", accuracy)  # Imprimir la precisión del clasificador
