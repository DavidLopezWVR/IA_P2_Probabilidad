#21110344  David López Rojas  6E2

import nltk
from nltk.corpus import treebank  # Importar el corpus Treebank, que contiene oraciones anotadas
from nltk.grammar import PCFG  # Importar la clase PCFG para gramáticas probabilísticas
from nltk.probability import MLEProbDist  # Importar la distribución de probabilidad MLE

# Descargar el corpus Treebank de NLTK (necesario solo la primera vez)
nltk.download('treebank')  # Descargar el corpus Treebank

# Obtener las producciones de las frases del corpus Treebank
productions = []  # Inicializar una lista para almacenar las producciones
for tree in treebank.parsed_sents():  # Iterar sobre las oraciones analizadas en el corpus
    productions += tree.productions()  # Añadir las producciones de cada árbol a la lista

# Crear una gramática PCFG utilizando las producciones
pcfg = PCFG.from_productions(productions)  # Crear una gramática probabilística a partir de las producciones

# Entrenar un modelo de distribución de probabilidad MLE para la gramática PCFG
mle_prob_dist = MLEProbDist(pcfg)  # Crear un modelo de estimación de máxima verosimilitud (MLE)

# Ejemplo: calcular la probabilidad de una producción específica
production = nltk.grammar.Production(nltk.Nonterminal('NP'), ['DT', 'NN'])  # Definir una producción para un sintagma nominal (NP)
print("Probabilidad de la producción", production, ":", mle_prob_dist.prob(production))  # Imprimir la probabilidad de la producción
