#21110344  David López Rojas  6E2

# Importar las clases necesarias de la biblioteca pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana, donde:
# - 'A' y 'B' son padres de 'C'.
# - 'C' es padre de 'D'.
model = BayesianNetwork([('A', 'C'), ('B', 'C'), ('C', 'D')])

# Definir las tablas de probabilidad condicional (CPDs) para cada nodo.

# CPD para el nodo 'A' (sin dependencias)
# A tiene dos posibles valores (0 y 1), con probabilidades de 0.3 y 0.7
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.3], [0.7]])

# CPD para el nodo 'B' (sin dependencias)
# B tiene dos posibles valores (0 y 1), con probabilidades de 0.6 y 0.4
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.6], [0.4]])

# CPD para el nodo 'C', condicionado a 'A' y 'B'
# La tabla de probabilidades tiene 4 combinaciones posibles de valores de A y B
# Las probabilidades especificadas son:
# P(C=0 | A=0, B=0) = 0.8, P(C=0 | A=0, B=1) = 0.4, etc.
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.4, 0.7, 0.1],
                           [0.2, 0.6, 0.3, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])

# CPD para el nodo 'D', condicionado a 'C'
# La tabla de probabilidades define P(D=0 | C=0) = 0.9 y P(D=0 | C=1) = 0.2, etc.
cpd_d = TabularCPD(variable='D', variable_card=2, 
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['C'], evidence_card=[2])

# Añadir las CPDs al modelo
model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)

# Verificar si el modelo es válido (consistente)
assert model.check_model()

# Crear un objeto de inferencia utilizando eliminación de variables
inference = VariableElimination(model)

# Calcular la probabilidad de 'D' dado que A=0 y B=1
posterior = inference.query(variables=['D'], evidence={'A': 0, 'B': 1})

# Imprimir el resultado de la probabilidad posterior
print("La probabilidad de D dado A=0 y B=1 es:", posterior.values[1])
