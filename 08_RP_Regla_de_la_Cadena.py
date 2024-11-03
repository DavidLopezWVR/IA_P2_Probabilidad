#21110344  David López Rojas  6E2

# Probabilidades marginales de los eventos A, B y C
P_A = 0.6  # Probabilidad de que ocurra el evento A
P_B_given_A = 0.7  # Probabilidad de que ocurra el evento B dado que A ha ocurrido
P_C_given_A_and_B = 0.8  # Probabilidad de que ocurra el evento C dado que A y B han ocurrido

# Calcular la probabilidad conjunta utilizando la Regla de la Cadena
# La regla de la cadena establece que la probabilidad conjunta de eventos A, B y C se puede calcular
# multiplicando la probabilidad del primer evento por la probabilidad del segundo evento dado el primero,
# y por la probabilidad del tercer evento dado los dos primeros.
P_A_and_B_and_C = P_A * P_B_given_A * P_C_given_A_and_B

# Imprimir el resultado
print("La probabilidad de que ocurran los eventos A, B y C en sucesión es:", P_A_and_B_and_C)
