#21110344  David López Rojas  6E2

# Importar la biblioteca pgmpy para trabajar con redes bayesianas
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana
model = BayesianNetwork([('A', 'C'), ('B', 'C')])  # A y B influyen en C

# Definir las tablas de probabilidad condicional (CPDs)
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.3], [0.7]])  # P(A)
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.6], [0.4]])  # P(B)

# Definir CPD de la variable C dado A y B
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.4, 0.7, 0.1],  # P(C=0 | A, B)
                           [0.2, 0.6, 0.3, 0.9]], # P(C=1 | A, B)
                   evidence=['A', 'B'], evidence_card=[2, 2])  # C depende de A y B

# Añadir las CPDs al modelo
model.add_cpds(cpd_a, cpd_b, cpd_c)

# Verificar si el modelo es válido
assert model.check_model()  # Comprueba la coherencia del modelo

# Realizar inferencia por enumeración para calcular la probabilidad de C dado A=0 y B=1
inference = VariableElimination(model)  # Inicializar el objeto de inferencia
posterior = inference.query(variables=['C'], evidence={'A': 0, 'B': 1})  # Calcular P(C | A=0, B=1)

# Imprimir la probabilidad de que C sea igual a 1 dado A=0 y B=1
print("La probabilidad de C dado A=0 y B=1 es:", posterior.values[1])  # Imprime P(C=1 | A=0, B=1)
