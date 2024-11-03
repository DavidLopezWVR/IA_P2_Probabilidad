#21110344  David López Rojas  6E2

import numpy as np

# Definición de la clase NaiveBayesClassifier
class NaiveBayesClassifier:
    def __init__(self):
        # Inicializa los parámetros del clasificador
        self.class_priors = None  # Probabilidades a priori de cada clase
        self.means = None         # Medias de cada característica por clase
        self.stds = None          # Desviaciones estándar de cada característica por clase

    # Método para ajustar el clasificador a los datos de entrenamiento
    def fit(self, X, y):
        # Calcular la probabilidad a priori de cada clase
        self.class_priors = {}
        for label in np.unique(y):
            # Calcula la proporción de cada clase en los datos
            self.class_priors[label] = np.mean(y == label)

        # Calcular las medias y desviaciones estándar de cada característica para cada clase
        self.means = {}
        self.stds = {}
        for label in np.unique(y):
            # Para cada clase, calcula la media y la desviación estándar de las características
            self.means[label] = np.mean(X[y == label], axis=0)
            self.stds[label] = np.std(X[y == label], axis=0)

    # Método para calcular la verosimilitud de una característica dada la media y desviación estándar
    def _calculate_likelihood(self, x, mean, std):
        # Fórmula de la distribución normal para calcular la verosimilitud
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    # Método para calcular la probabilidad posterior para cada clase
    def _calculate_posterior(self, X):
        posteriors = []  # Lista para almacenar las probabilidades posteriores
        for label in self.class_priors:
            prior = self.class_priors[label]  # Probabilidad a priori de la clase
            # Calcula la verosimilitud para cada característica y multiplica para obtener la probabilidad posterior
            likelihood = np.prod(self._calculate_likelihood(X, self.means[label], self.stds[label]), axis=1)
            posterior = prior * likelihood  # Multiplica la probabilidad a priori por la verosimilitud
            posteriors.append(posterior)  # Agrega la probabilidad posterior a la lista
        return np.array(posteriors).T  # Transpone para facilitar el acceso

    # Método para predecir la clase de nuevas instancias
    def predict(self, X):
        posteriors = self._calculate_posterior(X)  # Calcula las probabilidades posteriores
        return np.argmax(posteriors, axis=1)  # Devuelve el índice de la clase con la mayor probabilidad

# Datos de ejemplo (longitud y ancho del pétalo de flores)
X = np.array([[1.5, 0.3],  # Ejemplo de características (longitud y ancho)
              [4.5, 1.3],
              [5.7, 2.4],
              [1.3, 0.2],
              [5.2, 2.3]])
y = np.array([0, 1, 2, 0, 2])  # Etiquetas de clase (0, 1, 2)

# Crear y ajustar el clasificador Naive Bayes
classifier = NaiveBayesClassifier()
classifier.fit(X, y)  # Ajusta el clasificador a los datos de entrenamiento

# Datos de prueba
X_test = np.array([[1.4, 0.2],  # Nuevos ejemplos para predecir
                   [4.9, 2.0]])

# Predicciones
predictions = classifier.predict(X_test)  # Realiza predicciones sobre los datos de prueba
print("Predicciones:", predictions)  # Imprime las predicciones
