#21110344  David López Rojas  6E2

import cv2  # Importar la biblioteca OpenCV para el procesamiento de imágenes
import numpy as np  # Importar NumPy para manipulación de matrices
import tensorflow as tf  # Importar TensorFlow para modelos de aprendizaje profundo
from object_detection.utils import label_map_util  # Importar funciones útiles para manejar mapas de etiquetas

# Cargar el modelo preentrenado de detección de objetos de TensorFlow
model_path = 'ssd_mobilenet_v2_coco'  # Especificar la ruta del modelo preentrenado
model = tf.saved_model.load(model_path)  # Cargar el modelo guardado en la ruta especificada

# Cargar el archivo de configuración del mapa de etiquetas
label_map_path = 'mscoco_label_map.pbtxt'  # Especificar la ruta del archivo de mapa de etiquetas
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)  # Crear un índice de categorías a partir del mapa de etiquetas

# Función para realizar la detección de objetos en una imagen
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)  # Convertir la imagen a un tensor de TensorFlow
    input_tensor = input_tensor[tf.newaxis, ...]  # Agregar una dimensión adicional para el lote

    detections = model(input_tensor)  # Realizar la detección de objetos utilizando el modelo

    num_detections = int(detections.pop('num_detections'))  # Obtener el número de detecciones
    detections = {key: value[0, :num_detections].numpy()  # Convertir los resultados a formato NumPy
                   for key, value in detections.items()}  # Mantener solo las detecciones relevantes
    detections['num_detections'] = num_detections  # Almacenar el número de detecciones

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)  # Convertir las clases de detección a enteros

    return detections  # Retornar las detecciones

# Cargar una imagen de prueba
image_path = 'imagen.jpg'  # Especificar la ruta de la imagen de entrada
image = cv2.imread(image_path)  # Leer la imagen utilizando OpenCV
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR a RGB

# Realizar la detección de objetos en la imagen
detections = detect_objects(image_rgb)  # Llamar a la función de detección de objetos

# Dibujar los cuadros delimitadores de los objetos detectados en la imagen
for i in range(len(detections['detection_scores'])):  # Iterar sobre las puntuaciones de detección
    class_id = int(detections['detection_classes'][i])  # Obtener el ID de la clase de la detección
    score = detections['detection_scores'][i]  # Obtener la puntuación de la detección
    if score > 0.5:  # Filtrar detecciones con puntuaciones menores a 0.5
        label = category_index[class_id]['name']  # Obtener la etiqueta de la clase
        bbox = detections['detection_boxes'][i]  # Obtener las coordenadas de la caja delimitadora
        h, w, _ = image.shape  # Obtener las dimensiones de la imagen original
        ymin, xmin, ymax, xmax = bbox  # Descomponer las coordenadas de la caja delimitadora
        # Convertir las coordenadas normalizadas a píxeles
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        # Dibujar un rectángulo verde alrededor del objeto detectado
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Colocar el texto con la etiqueta y la puntuación de la detección
        cv2.putText(image, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con los objetos detectados
cv2.imshow('Detected Objects', image)  # Mostrar la imagen con los cuadros delimitadores
cv2.waitKey(0)  # Esperar a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
