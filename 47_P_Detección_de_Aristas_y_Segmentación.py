#21110344  David Lopez Rojas  6E2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes
import numpy as np  # Importar NumPy para manipulaciones numéricas y matrices
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización de imágenes

# Cargar una imagen en escala de grises
image = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)  # Leer la imagen y cargarla en escala de grises

# Aplicar el operador de Sobel para detección de bordes
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Aplicar el operador de Sobel en la dirección X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Aplicar el operador de Sobel en la dirección Y
edges = np.sqrt(sobel_x**2 + sobel_y**2)  # Combinar las imágenes de Sobel en X e Y para obtener la magnitud de los bordes

# Aplicar umbralización para obtener una imagen binaria de los bordes
edges_binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)[1]  # Umbralizar la magnitud de los bordes para obtener una imagen binaria

# Realizar la segmentación utilizando la umbralización de Otsu
_, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Umbralización de Otsu para segmentar la imagen original

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 6))  # Crear una figura con tamaño 12x6 para la visualización

# Subgráfica para la imagen original
plt.subplot(2, 2, 1)  # Crear un subplot en una cuadrícula de 2x2, posición 1
plt.imshow(image, cmap='gray')  # Mostrar la imagen original en escala de grises
plt.title('Imagen Original')  # Establecer el título del subplot

# Subgráfica para la detección de bordes
plt.subplot(2, 2, 2)  # Crear un subplot en posición 2
plt.imshow(edges, cmap='gray')  # Mostrar la imagen de detección de bordes
plt.title('Detección de Bordes (Sobel)')  # Título para esta subgráfica

# Subgráfica para los bordes binarios
plt.subplot(2, 2, 3)  # Crear un subplot en posición 3
plt.imshow(edges_binary, cmap='gray')  # Mostrar la imagen binaria de los bordes
plt.title('Bordes Binarios')  # Título para esta subgráfica

# Subgráfica para la segmentación
plt.subplot(2, 2, 4)  # Crear un subplot en posición 4
plt.imshow(segmented_image, cmap='gray')  # Mostrar la imagen segmentada
plt.title('Segmentación (Umbralización de Otsu)')  # Título para esta subgráfica

plt.tight_layout()  # Ajustar el espaciado entre subgráficas para que no se superpongan
plt.show()  # Mostrar todas las subgráficas en una ventana
