#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para la manipulación de matrices
import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes

# Crear una imagen de fondo
background = np.zeros((400, 400, 3), dtype=np.uint8)  # Crear una matriz de ceros con dimensiones 400x400 y 3 canales (RGB)
background[:] = (255, 255, 255)  # Asignar el color blanco (255, 255, 255) a toda la imagen como fondo

# Dibujar una textura (patrón de líneas diagonales)
for i in range(0, 400, 20):  # Iterar desde 0 hasta 400 con un paso de 20
    cv2.line(background, (0, i), (400, i), (0, 0, 0), thickness=2)  # Dibujar líneas horizontales negras
    cv2.line(background, (i, 0), (i, 400), (0, 0, 0), thickness=2)  # Dibujar líneas verticales negras

# Crear una máscara para la sombra
mask = np.zeros((400, 400), dtype=np.uint8)  # Crear una matriz de ceros para la máscara, en escala de grises
cv2.circle(mask, (200, 200), 100, (255, 255, 255), thickness=-1)  # Dibujar un círculo blanco en la máscara

# Crear una imagen con sombra
shadow = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)  # Convertir la imagen de fondo a escala de grises
shadow = cv2.bitwise_and(shadow, mask)  # Aplicar la máscara a la imagen de sombra, conservando solo la región del círculo

# Mostrar las imágenes resultantes
cv2.imshow('Textura', background)  # Mostrar la imagen de fondo con el patrón de textura
cv2.imshow('Sombra', shadow)  # Mostrar la imagen de sombra
cv2.waitKey(0)  # Esperar a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
