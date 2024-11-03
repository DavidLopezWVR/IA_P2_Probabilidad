#21110344  David López Rojas  6E2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes
import numpy as np  # Importar NumPy para manipulación de matrices (no se utiliza directamente en este código)
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización de imágenes

# Cargar una imagen
image = cv2.imread('imagen.jpg')  # Leer la imagen desde un archivo. 'imagen.jpg' debe estar en el mismo directorio o especificar la ruta.

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir la imagen cargada de BGR a escala de grises

# Aplicar un filtro de suavizado Gaussiano
gaussian_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Aplicar un filtro de suavizado Gaussiano con un kernel de 5x5

# Aplicar un filtro de mediana
median_blurred = cv2.medianBlur(gray_image, 5)  # Aplicar un filtro de mediana con un tamaño de kernel de 5

# Aplicar un filtro bilateral
bilateral_filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)  # Aplicar un filtro bilateral con un tamaño de radio de 9 y parámetros de sigma para el color y la distancia de 75

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 6))  # Crear una figura de tamaño 12x6 para la visualización

# Subgráfica para la imagen en escala de grises
plt.subplot(2, 2, 1)  # Crear un subplot en una cuadrícula de 2x2, posición 1
plt.imshow(gray_image, cmap='gray')  # Mostrar la imagen en escala de grises
plt.title('Imagen en escala de grises')  # Establecer el título del subplot

# Subgráfica para el suavizado gaussiano
plt.subplot(2, 2, 2)  # Crear un subplot en posición 2
plt.imshow(gaussian_blurred, cmap='gray')  # Mostrar la imagen suavizada con filtro Gaussiano
plt.title('Suavizado Gaussiano')  # Título para esta subgráfica

# Subgráfica para el suavizado mediano
plt.subplot(2, 2, 3)  # Crear un subplot en posición 3
plt.imshow(median_blurred, cmap='gray')  # Mostrar la imagen suavizada con filtro de mediana
plt.title('Suavizado Mediano')  # Título para esta subgráfica

# Subgráfica para el filtro bilateral
plt.subplot(2, 2, 4)  # Crear un subplot en posición 4
plt.imshow(bilateral_filtered, cmap='gray')  # Mostrar la imagen procesada con filtro bilateral
plt.title('Filtro Bilateral')  # Título para esta subgráfica

plt.tight_layout()  # Ajustar el espaciado entre subgráficas para que no se superpongan
plt.show()  # Mostrar todas las subgráficas en una ventana
