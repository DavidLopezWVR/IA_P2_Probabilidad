#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para manipulación de arreglos y operaciones matemáticas

class SMT1Translator:
    def __init__(self, source_vocab, target_vocab):
        # Inicializar el traductor con vocabularios de origen y destino
        self.source_vocab = source_vocab  # Vocabulario de palabras en el idioma de origen
        self.target_vocab = target_vocab  # Vocabulario de palabras en el idioma de destino
        # Inicializar las probabilidades de traducción de manera aleatoria
        self.translation_probs = np.random.rand(len(source_vocab), len(target_vocab))

    def translate(self, source_sentence):
        # Dividir la oración de origen en palabras y obtener sus índices
        source_indices = [self.source_vocab.index(word) for word in source_sentence.split()]
        # Obtener los índices de las palabras traducidas utilizando la probabilidad máxima
        target_indices = np.argmax(self.translation_probs[source_indices], axis=1)
        # Convertir los índices de las palabras de destino de nuevo a palabras
        target_sentence = ' '.join([self.target_vocab[index] for index in target_indices])
        return target_sentence  # Devolver la oración traducida

# Ejemplo de uso
source_vocab = ['cat', 'dog', 'house']  # Definición del vocabulario de origen
target_vocab = ['gato', 'perro', 'casa']  # Definición del vocabulario de destino

translator = SMT1Translator(source_vocab, target_vocab)  # Crear una instancia del traductor

source_sentence = 'cat house'  # Definir una oración de origen para traducir
target_sentence = translator.translate(source_sentence)  # Traducir la oración de origen
# Imprimir el resultado de la traducción
print("Traducción de '{}': {}".format(source_sentence, target_sentence))
