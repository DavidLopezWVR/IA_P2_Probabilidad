#21110344  David López Rojas  6E2

import speech_recognition as sr  # Importar la biblioteca de reconocimiento de voz

# Crear un objeto reconocedor
recognizer = sr.Recognizer()  # Inicializar un objeto reconocedor que se utilizará para reconocer el habla

# Configurar el micrófono como fuente de entrada
microphone = sr.Microphone()  # Inicializar un objeto de micrófono para capturar el audio

# Escuchar del micrófono
with microphone as source:  # Usar el micrófono como fuente de audio
    print("Di algo...")  # Pedir al usuario que hable
    audio = recognizer.listen(source)  # Escuchar y guardar el audio capturado en la variable 'audio'

# Realizar el reconocimiento de voz
try:
    print("Reconociendo...")  # Indicar que se está procesando el reconocimiento
    # Utilizar la API de Google para convertir el audio a texto en español
    text = recognizer.recognize_google(audio, language="es-ES")  
    print("Texto reconocido:", text)  # Mostrar el texto que ha sido reconocido
except sr.UnknownValueError:
    # Manejar el error cuando el reconocimiento no puede entender el audio
    print("No se pudo entender el habla")  
except sr.RequestError as e:
    # Manejar el error cuando hay un problema con la solicitud a la API de Google
    print("Error al solicitar resultados; {0}".format(e))  
