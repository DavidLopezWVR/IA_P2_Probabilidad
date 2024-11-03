#21110344  David López Rojas  6E2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes
import pytesseract  # Importar la biblioteca Tesseract para el reconocimiento de texto

# Configuración de Tesseract (ubicación del ejecutable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Especificar la ruta del ejecutable de Tesseract en el sistema

# Cargar la imagen
image = cv2.imread('handwritten_text.png')  # Leer la imagen que contiene texto manuscrito

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir la imagen de BGR a escala de grises para simplificar el procesamiento

# Aplicar umbral (binarización) para resaltar los caracteres
_, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarizar la imagen utilizando el método de Otsu para separar el fondo del texto

# Realizar OCR en la imagen
text = pytesseract.image_to_string(threshold_image, lang='eng')  # Aplicar Tesseract para reconocer el texto en la imagen binarizada, especificando que el idioma es inglés

# Mostrar el texto reconocido
print("Texto reconocido:")  # Imprimir encabezado para el texto reconocido
print(text)  # Imprimir el texto que ha sido reconocido por Tesseract
