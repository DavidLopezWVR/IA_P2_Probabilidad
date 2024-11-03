#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para manipulaciones de arreglos y generación de números aleatorios
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización de datos

# Crear una matriz de píxeles (imagen) con valores aleatorios entre 0 y 255
image = np.random.randint(0, 256, size=(100, 100))
# Se genera una matriz de 100x100 donde cada elemento es un número entero aleatorio en el rango de 0 a 255,
# que representa un píxel en una imagen en escala de grises (0 es negro y 255 es blanco).

# Mostrar la imagen utilizando matplotlib
plt.imshow(image, cmap='gray')  # Mostrar la matriz como una imagen en escala de grises
plt.axis('off')  # Ocultar ejes para que la imagen se muestre sin marcas de ejes
plt.title('Imagen generada')  # Título de la figura que se mostrará en la parte superior
plt.show()  # Mostrar la figura con la imagen generada
