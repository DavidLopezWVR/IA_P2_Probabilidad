#21110344  David López Rojas  6E2

import pygame  # Importar la biblioteca Pygame para crear juegos y gráficos
import random  # Importar la biblioteca random para generar movimientos aleatorios

# Inicializar pygame
pygame.init()  # Inicializar todos los módulos de Pygame

# Definir colores
WHITE = (255, 255, 255)  # Color blanco
BLACK = (0, 0, 0)  # Color negro
RED = (255, 0, 0)  # Color rojo

# Definir tamaño de la ventana
WIDTH, HEIGHT = 800, 600  # Ancho y alto de la ventana del juego

# Clase para el robot
class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()  # Llamar al constructor de la clase base Sprite
        self.image = pygame.Surface((50, 50))  # Crear una superficie de 50x50 píxeles para el robot
        self.image.fill(RED)  # Rellenar la superficie con color rojo
        self.rect = self.image.get_rect()  # Obtener el rectángulo que delimita la imagen
        self.rect.center = (x, y)  # Establecer la posición inicial del robot en el centro de las coordenadas (x, y)
        self.speed = 3  # Establecer la velocidad de movimiento del robot

    def update(self):
        # Movimiento aleatorio del robot
        direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])  # Elegir una dirección de movimiento aleatoria
        if direction == 'UP':
            self.rect.y -= self.speed  # Mover hacia arriba
        elif direction == 'DOWN':
            self.rect.y += self.speed  # Mover hacia abajo
        elif direction == 'LEFT':
            self.rect.x -= self.speed  # Mover hacia la izquierda
        elif direction == 'RIGHT':
            self.rect.x += self.speed  # Mover hacia la derecha

        # Verificar límites de la ventana
        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))  # Mantener el robot dentro del ancho de la ventana
        self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))  # Mantener el robot dentro de la altura de la ventana

# Configurar la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Crear la ventana con el tamaño definido
pygame.display.set_caption("HW Robótico: Sensores y Actuadores")  # Establecer el título de la ventana

clock = pygame.time.Clock()  # Crear un objeto Clock para controlar la tasa de actualización de la pantalla

# Crear el robot
robot = Robot(WIDTH // 2, HEIGHT // 2)  # Instanciar un objeto Robot en el centro de la ventana

# Crear grupo de sprites y agregar el robot
all_sprites = pygame.sprite.Group()  # Crear un grupo para contener todos los sprites
all_sprites.add(robot)  # Agregar el robot al grupo de sprites

# Bucle principal
running = True  # Variable para controlar el bucle del juego
while running:
    # Manejo de eventos
    for event in pygame.event.get():  # Procesar eventos en la cola
        if event.type == pygame.QUIT:  # Si se cierra la ventana
            running = False  # Terminar el bucle

    # Actualizar
    all_sprites.update()  # Actualizar todos los sprites en el grupo

    # Renderizar
    screen.fill(WHITE)  # Rellenar la pantalla con color blanco
    all_sprites.draw(screen)  # Dibujar todos los sprites en la pantalla

    # Refrescar la pantalla
    pygame.display.flip()  # Actualizar la ventana para mostrar los cambios

    # Controlar la velocidad de la actualización
    clock.tick(60)  # Limitar el bucle a 60 cuadros por segundo

# Salir del programa
pygame.quit()  # Cerrar Pygame y liberar los recursos utilizados
