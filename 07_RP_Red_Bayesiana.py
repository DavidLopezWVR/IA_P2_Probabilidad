#21110344  David López Rojas  6E2

# Importar clases necesarias de la biblioteca pgmpy
from pgmpy.models import BayesianNetwork  # Para crear la red bayesiana
from pgmpy.factors.discrete import TabularCPD  # Para definir las distribuciones de probabilidad condicional
from pgmpy.inference import VariableElimination  # Para realizar inferencias en la red

# Definir la estructura de la red bayesiana
# Se establece una relación de dependencia entre las variables D (causa) y S1, S2 (efectos)
model = BayesianNetwork([('D', 'S1'), ('D', 'S2')])

# Definir las tablas de probabilidad condicional (CPDs)
# CPD para la variable D
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6], [0.4]])  
# D tiene dos estados: 0 (no ocurre) con probabilidad 0.6 y 1 (ocurre) con probabilidad 0.4.

# CPD para la variable S1, que depende de D
cpd_s1 = TabularCPD(variable='S1', variable_card=2, 
                    values=[[0.8, 0.2],  # Probabilidades de S1 dado D = 0
                            [0.2, 0.8]], # Probabilidades de S1 dado D = 1
                    evidence=['D'], evidence_card=[2])
# Si D es 0, hay un 80% de probabilidad de que S1 sea 0 y un 20% de probabilidad de que S1 sea 1.
# Si D es 1, hay un 20% de probabilidad de que S1 sea 0 y un 80% de probabilidad de que S1 sea 1.

# CPD para la variable S2, que también depende de D
cpd_s2 = TabularCPD(variable='S2', variable_card=2, 
                    values=[[0.7, 0.3],  # Probabilidades de S2 dado D = 0
                            [0.3, 0.7]], # Probabilidades de S2 dado D = 1
                    evidence=['D'], evidence_card=[2])
# La interpretación es similar a la de S1: si D es 0, S2 tiene un 70% de probabilidad de ser 0 y un 30% de ser 1,
# mientras que si D es 1, S2 tiene un 30% de probabilidad de ser 0 y un 70% de probabilidad de ser 1.

# Añadir las CPDs al modelo
model.add_cpds(cpd_d, cpd_s1, cpd_s2)

# Verificar si el modelo es válido
assert model.check_model()  # Asegura que la estructura de la red y las CPDs sean consistentes

# Realizar inferencia probabilística
inference = VariableElimination(model)  # Crear un objeto para realizar inferencia mediante eliminación de variables
posterior = inference.query(variables=['D'], evidence={'S1': 1, 'S2': 0})  # Realizar la consulta: dado que S1=1 y S2=0, calcular la probabilidad posterior de D
print(posterior['D'])  # Imprimir la distribución de probabilidad de D
