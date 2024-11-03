#21110344  David López Rojas  6E2

import numpy as np

# Función para verificar la independencia condicional
def check_conditional_independence(data, var1, var2, condition):
    # Filtrar los datos que cumplen con la condición especificada
    filtered_data = data[data[:, 2] == condition]
    
    # Calcular las probabilidades condicionales de var1 y var2
    p_var1_given_condition = np.mean(filtered_data[:, 0] == var1)  # Probabilidad de var1 dado la condición
    p_var2_given_condition = np.mean(filtered_data[:, 1] == var2)  # Probabilidad de var2 dado la condición
    
    # Calcular la probabilidad conjunta de var1 y var2 dado la condición
    p_joint = np.mean((filtered_data[:, 0] == var1) & (filtered_data[:, 1] == var2))
    
    # Verificar la independencia condicional
    if np.isclose(p_joint, p_var1_given_condition * p_var2_given_condition):
        return True  # Las variables son independientes condicionalmente
    else:
        return False  # Las variables no son independientes condicionalmente

# Generar datos aleatorios: dos variables (var1 y var2) y una variable de condición
np.random.seed(42)  # Fijar la semilla para reproducibilidad
data_size = 1000  # Tamaño del conjunto de datos
var1_values = np.random.choice([0, 1], size=data_size)  # Valores aleatorios para var1
var2_values = np.random.choice([0, 1], size=data_size)  # Valores aleatorios para var2
condition_values = np.random.choice([0, 1], size=data_size)  # Valores aleatorios para la condición

# Combinar los valores en un solo conjunto de datos
data = np.column_stack((var1_values, var2_values, condition_values))  # Crear un array de datos

# Verificar la independencia condicional dada la condición
var1 = 0  # Valor de la primera variable para la prueba
var2 = 0  # Valor de la segunda variable para la prueba
condition = 0  # Valor de la condición para la prueba
is_independent = check_conditional_independence(data, var1, var2, condition)  # Llamada a la función

# Imprimir el resultado
print(f"¿Las variables {var1} y {var2} son independientes condicionales a la condición {condition}? {is_independent}")
