#21110344  David López Rojas  6E2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes y video

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)  # Inicializar la captura de video desde la cámara por defecto

# Inicializar el primer frame
_, frame1 = cap.read()  # Leer el primer frame del video
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convertir el primer frame a escala de grises

# Configurar el detector de movimiento
motion_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)  # Inicializar el detector de movimiento usando MOG2

while True:
    # Capturar el siguiente frame
    _, frame2 = cap.read()  # Leer el siguiente frame del video
    current_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # Convertir el frame actual a escala de grises

    # Obtener la máscara de movimiento
    mask = motion_detector.apply(frame2)  # Aplicar el detector de movimiento al frame actual

    # Filtrar el ruido y encontrar contornos
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  # Aplicar umbralización para binarizar la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontrar contornos en la máscara binarizada

    # Dibujar rectángulos alrededor de los objetos en movimiento
    for contour in contours:  # Iterar sobre cada contorno detectado
        x, y, w, h = cv2.boundingRect(contour)  # Obtener el rectángulo delimitador del contorno
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar un rectángulo verde alrededor del objeto en movimiento

    # Mostrar el frame con los objetos en movimiento
    cv2.imshow('Motion Detection', frame2)  # Mostrar el frame actual con las detecciones en una ventana

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Verificar si se presiona la tecla 'q'
        break  # Salir del bucle si se presiona 'q'

# Liberar la captura y cerrar la ventana
cap.release()  # Liberar el objeto de captura de video
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
