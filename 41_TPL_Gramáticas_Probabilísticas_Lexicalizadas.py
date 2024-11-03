#21110344  David López Rojas  6E2

import nltk
from nltk.corpus import treebank  # Importar el corpus Treebank, que contiene oraciones analizadas
from nltk.grammar import PCFG, induce_pcfg  # Importar la clase PCFG y la función para inducir PCFG
from nltk.probability import MLEProbDist  # Importar la distribución de probabilidad MLE

# Descargar el corpus Treebank de NLTK (necesario solo la primera vez)
nltk.download('treebank')  # Descargar el corpus Treebank

# Obtener el corpus Treebank
corpus = treebank.parsed_sents()  # Obtener las oraciones analizadas del corpus

# Extraer las producciones binarias del corpus Treebank
binary_productions = []  # Inicializar una lista para almacenar las producciones binarias
for tree in corpus:  # Iterar sobre los árboles de análisis en el corpus
    binary_productions += tree.productions()  # Añadir las producciones de cada árbol a la lista

# Inducir una gramática PCFG lexicalizada a partir de las producciones binarias
lpcfg = induce_pcfg(nltk.Nonterminal('S'), binary_productions)  # Inducir la LPCFG a partir de las producciones

# Entrenar un modelo de distribución de probabilidad MLE para la gramática LPCFG
mle_prob_dist = MLEProbDist(lpcfg)  # Crear un modelo de estimación de máxima verosimilitud (MLE)

# Ejemplo: calcular la probabilidad de una producción específica
production = nltk.grammar.Production(nltk.Nonterminal('NP'), ['DT', 'NN'])  # Definir una producción para un sintagma nominal (NP)
print("Probabilidad de la producción", production, ":", mle_prob_dist.prob(production))  # Imprimir la probabilidad de la producción
