#21110344  David López Rojas  6E2

import numpy as np  # Importar NumPy para realizar operaciones numéricas
import matplotlib.pyplot as plt  # Importar Matplotlib para visualizar gráficos

# Función para verificar si una configuración es válida en el entorno
def is_valid_configuration(configuration):
    x, y = configuration  # Descomponer la configuración en sus componentes x e y
    # En este ejemplo, el robot no puede ir más allá de ciertos límites en el entorno
    return 0 <= x <= 10 and 0 <= y <= 10  # Retorna True si la configuración está dentro de los límites, de lo contrario False

# Función para visualizar el espacio de configuración
def visualize_configuration_space():
    # Definir límites del espacio de configuración
    x_range = np.arange(0, 11, 0.1)  # Crear un rango para las posiciones en x desde 0 a 10 en incrementos de 0.1
    y_range = np.arange(0, 11, 0.1)  # Crear un rango para las posiciones en y desde 0 a 10 en incrementos de 0.1
    
    # Crear una cuadrícula de configuraciones y verificar su validez
    configuration_space = []  # Inicializar una lista para almacenar configuraciones válidas
    for x in x_range:  # Iterar sobre cada valor de x en el rango definido
        for y in y_range:  # Iterar sobre cada valor de y en el rango definido
            configuration = [x, y]  # Crear una configuración de dos dimensiones
            if is_valid_configuration(configuration):  # Comprobar si la configuración es válida
                configuration_space.append(configuration)  # Agregar la configuración válida a la lista
    
    # Convertir a matriz para trazado
    configuration_space = np.array(configuration_space)  # Convertir la lista de configuraciones válidas a un array de NumPy
    
    # Visualizar el espacio de configuración
    plt.figure(figsize=(8, 6))  # Crear una nueva figura con un tamaño específico
    plt.scatter(configuration_space[:, 0], configuration_space[:, 1], s=1, color='blue')  # Graficar las configuraciones válidas
    plt.xlabel('Posición en X')  # Etiqueta para el eje X
    plt.ylabel('Posición en Y')  # Etiqueta para el eje Y
    plt.title('Espacio de Configuración')  # Título del gráfico
    plt.grid(True)  # Activar la cuadrícula en el gráfico
    plt.axis('equal')  # Asegurar que las proporciones del gráfico sean iguales
    plt.show()  # Mostrar el gráfico

# Visualizar el espacio de configuración
visualize_configuration_space()  # Llamar a la función para generar y mostrar el espacio de configuración
