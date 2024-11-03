#21110344  David López Rojas  6E2

import numpy as np  # Importar la biblioteca NumPy para cálculos numéricos
import matplotlib.pyplot as plt  # Importar matplotlib para visualización

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        # Inicializar el controlador PID con las ganancias proporcionadas
        self.Kp = Kp  # Ganancia proporcional
        self.Ki = Ki  # Ganancia integral
        self.Kd = Kd  # Ganancia derivativa
        self.prev_error = 0  # Error anterior (para el cálculo derivativo)
        self.integral = 0  # Integral acumulada del error
    
    def control(self, setpoint, current_value, dt):
        # Calcular la salida del controlador PID
        error = setpoint - current_value  # Calcular el error
        self.integral += error * dt  # Acumular el error en el tiempo
        derivative = (error - self.prev_error) / dt  # Calcular la derivada del error
        # Calcular la salida del controlador usando la fórmula PID
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error  # Actualizar el error previo para la próxima iteración
        return output  # Retornar la salida del controlador

class MassSpringDamper:
    def __init__(self, m, k, c):
        # Inicializar el sistema masa-resorte-amortiguador
        self.m = m  # Masa del sistema (kg)
        self.k = k  # Constante del resorte (N/m)
        self.c = c  # Coeficiente de amortiguación (Ns/m)
    
    def update(self, force, dt):
        # Actualizar la aceleración del sistema dado una fuerza aplicada
        a = force / self.m  # Aceleración = Fuerza / Masa
        return a  # Retornar la aceleración

# Parámetros del sistema
m = 1.0  # Masa (kg)
k = 10.0  # Constante del resorte (N/m)
c = 1.0   # Coeficiente de amortiguación (Ns/m)

# Crear objetos de controlador PID y sistema
controller = PIDController(Kp=20, Ki=5, Kd=10)  # Instancia del controlador PID con ganancias
system = MassSpringDamper(m, k, c)  # Instancia del sistema masa-resorte-amortiguador

# Parámetros de simulación
dt = 0.01  # Paso de tiempo de la simulación
total_time = 10.0  # Tiempo total de la simulación
num_steps = int(total_time / dt)  # Número total de pasos en la simulación

# Condiciones iniciales
current_position = 0.0  # Posición inicial del sistema (m)
current_velocity = 0.0  # Velocidad inicial del sistema (m/s)

# Listas para almacenar datos de la simulación
time_values = np.arange(0, total_time, dt)  # Crear un array de tiempo para la simulación
position_values = []  # Lista para almacenar las posiciones a lo largo del tiempo

# Simulación
for _ in range(num_steps):
    # Calcular la fuerza de control utilizando el controlador PID
    target_position = 1.0  # Posición deseada (setpoint)
    control_force = controller.control(target_position, current_position, dt)  # Fuerza de control calculada
    
    # Actualizar la posición y la velocidad del sistema utilizando la dinámica del sistema
    acceleration = system.update(control_force, dt)  # Obtener la aceleración del sistema
    current_velocity += acceleration * dt  # Actualizar la velocidad (v = v + a * dt)
    current_position += current_velocity * dt  # Actualizar la posición (x = x + v * dt)
    
    # Almacenar la posición actual para su visualización
    position_values.append(current_position)  # Añadir la posición actual a la lista de posiciones

# Visualizar resultados
plt.plot(time_values, position_values, label='Posición')  # Graficar la posición a lo largo del tiempo
plt.xlabel('Tiempo (s)')  # Etiqueta del eje X
plt.ylabel('Posición (m)')  # Etiqueta del eje Y
plt.title('Control PID de un sistema masa-resorte-amortiguador')  # Título del gráfico
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar la cuadrícula
plt.show()  # Mostrar el gráfico
