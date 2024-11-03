#21110344  David López Rojas  6E2

# Importar numpy para cálculos numéricos
import numpy as np

# Clase para el clasificador de Bayes
class BayesianClassifier:
    def __init__(self):
        # Inicializar las probabilidades a priori de las clases
        self.p_spam = None  # Probabilidad a priori de que un correo sea spam
        self.p_not_spam = None  # Probabilidad a priori de que un correo no sea spam

        # Inicializar los parámetros de las características
        self.mean_spam = None  # Media de las características para los correos spam
        self.std_spam = None  # Desviación estándar de las características para los correos spam
        self.mean_not_spam = None  # Media de las características para los correos no spam
        self.std_not_spam = None  # Desviación estándar de las características para los correos no spam

    # Entrenar el clasificador con datos de entrenamiento
    def train(self, X_train, y_train):
        # Calcular las probabilidades a priori de las clases
        self.p_spam = np.mean(y_train)  # Proporción de correos spam en los datos de entrenamiento
        self.p_not_spam = 1 - self.p_spam  # Proporción de correos no spam

        # Calcular las medias y desviaciones estándar de las características para cada clase
        self.mean_spam = np.mean(X_train[y_train == 1], axis=0)  # Media de las características para spam
        self.std_spam = np.std(X_train[y_train == 1], axis=0)  # Desviación estándar para spam
        self.mean_not_spam = np.mean(X_train[y_train == 0], axis=0)  # Media de las características para no spam
        self.std_not_spam = np.std(X_train[y_train == 0], axis=0)  # Desviación estándar para no spam

    # Predecir la clase de nuevos datos
    def predict(self, X):
        # Calcular las probabilidades condicionales de las características dadas las clases
        p_x_given_spam = np.prod(self._calculate_likelihood(X, self.mean_spam, self.std_spam), axis=1)
        p_x_given_not_spam = np.prod(self._calculate_likelihood(X, self.mean_not_spam, self.std_not_spam), axis=1)

        # Calcular las probabilidades a posteriori de las clases
        p_spam_given_x = p_x_given_spam * self.p_spam  # Probabilidad a posteriori de spam
        p_not_spam_given_x = p_x_given_not_spam * self.p_not_spam  # Probabilidad a posteriori de no spam

        # Clasificar los datos basados en las probabilidades a posteriori
        y_pred = p_spam_given_x > p_not_spam_given_x  # Comparar probabilidades a posteriori
        return y_pred.astype(int)  # Convertir booleanos a enteros (0 o 1)

    # Calcular la función de densidad de probabilidad (PDF) de una distribución normal
    def _calculate_likelihood(self, x, mean, std):
        # Calcular el exponente de la fórmula de la distribución normal
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        # Retornar la función de densidad de probabilidad
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Datos de entrenamiento (características y etiquetas)
X_train = np.array([[3, 10], [5, 15], [2, 8], [8, 20]])  # Características de los correos
y_train = np.array([1, 1, 0, 0])  # 1: spam, 0: no spam

# Datos de prueba
X_test = np.array([[4, 12], [6, 18]])  # Nuevos correos para clasificar

# Crear y entrenar el clasificador de Bayes
classifier = BayesianClassifier()  # Instancia del clasificador
classifier.train(X_train, y_train)  # Entrenar el clasificador con los datos de entrenamiento

# Predecir las etiquetas de los datos de prueba
y_pred = classifier.predict(X_test)  # Clasificación de los nuevos correos

# Imprimir las predicciones
print("Predicciones:", y_pred)  # Mostrar resultados de la clasificación
