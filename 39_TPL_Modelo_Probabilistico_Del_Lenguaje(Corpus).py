#21110344  David López Rojas  6E2

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import random

# Descargar el tokenizer y el corpus (necesario solo la primera vez)
nltk.download('punkt')  # Descarga el tokenizer para separar palabras
nltk.download('brown')   # Descarga el corpus Brown, un conjunto de textos en inglés

# Cargar el corpus Brown de NLTK
corpus = nltk.corpus.brown  # Acceder al corpus Brown

# Obtener las palabras del corpus
words = corpus.words()  # Extraer todas las palabras del corpus

# Tokenizar las palabras y calcular la distribución de frecuencia
word_freq = FreqDist(words)  # Calcular la distribución de frecuencia de las palabras

# Ejemplo de las 10 palabras más frecuentes en el corpus
print("10 palabras más frecuentes en el corpus:")
print(word_freq.most_common(10))  # Imprimir las 10 palabras más frecuentes y su frecuencia

# Función para generar texto aleatorio basado en el modelo probabilístico
def generate_text(model, num_words=50):
    generated_text = []  # Inicializar una lista para almacenar las palabras generadas
    for _ in range(num_words):  # Iterar el número de palabras que se desean generar
        word = model.generate()  # Generar una palabra usando el modelo
        generated_text.append(word)  # Añadir la palabra generada a la lista
    return ' '.join(generated_text)  # Unir las palabras en una cadena de texto

# Clase para el modelo probabilístico del lenguaje
class LanguageModel:
    def __init__(self, word_freq):
        self.words = list(word_freq.keys())  # Guardar las palabras como una lista
        self.probs = np.array(list(word_freq.values())) / sum(word_freq.values())  # Calcular las probabilidades normalizadas

    def generate(self):
        return random.choices(self.words, weights=self.probs)[0]  # Seleccionar una palabra basada en las probabilidades

# Crear el modelo probabilístico del lenguaje
model = LanguageModel(word_freq)  # Inicializar el modelo con la distribución de frecuencia de las palabras

# Generar texto aleatorio basado en el modelo
generated_text = generate_text(model, num_words=50)  # Generar 50 palabras aleatorias
print("\nTexto generado:")
print(generated_text)  # Imprimir el texto generado aleatorio
