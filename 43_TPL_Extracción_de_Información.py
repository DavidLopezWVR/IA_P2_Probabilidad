#21110344  David López Rojas  6E2

import nltk  # Importar la biblioteca NLTK para procesamiento de lenguaje natural
from nltk.tokenize import word_tokenize, sent_tokenize  # Importar funciones para tokenizar texto en palabras y oraciones
from nltk import pos_tag, ne_chunk  # Importar funciones para etiquetar partes del habla y para reconocimiento de entidades nombradas

# Ejemplo de texto
text = """
Barack Obama was born in Hawaii. He was the 44th President of the United States.
Joe Biden is the current President. He was Vice President under Obama.
"""

# Tokenizar el texto en oraciones y palabras
sentences = sent_tokenize(text)  # Dividir el texto en oraciones
words = [word_tokenize(sentence) for sentence in sentences]  # Dividir cada oración en palabras

# Etiquetar partes del habla (POS tagging) para cada palabra en cada oración
tagged_words = [pos_tag(sentence) for sentence in words]  # Etiquetar cada palabra con su parte del habla correspondiente

# Identificar entidades nombradas (NER) en el texto
named_entities = [ne_chunk(tagged_word) for tagged_word in tagged_words]  # Realizar reconocimiento de entidades nombradas en las palabras etiquetadas

# Extraer nombres de personas y sus roles
people_roles = []  # Lista para almacenar nombres y roles
for named_entity in named_entities:
    for chunk in named_entity:  # Recorrer cada entidad nombrada
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':  # Verificar si el chunk es una entidad de tipo PERSON
            person = ' '.join([c[0] for c in chunk])  # Unir las palabras de la entidad nombrada para formar el nombre
            role = ' '.join([c[0] for c in chunk.subtrees() if c.label() != 'PERSON'])  # Unir las palabras de los subárboles que no son PERSON para formar el rol
            people_roles.append((person, role))  # Agregar el nombre y rol a la lista

# Imprimir los nombres de personas y sus roles
for person, role in people_roles:
    print("Nombre:", person)  # Imprimir el nombre de la persona
    print("Rol:", role)  # Imprimir el rol asociado
    print()  # Línea en blanco para mayor claridad en la salida
