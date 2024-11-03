#21110344  David López Rojas  6E2

import heapq  # Importar la biblioteca heapq para usar una cola de prioridad

class Node:
    def __init__(self, x, y, parent=None):
        # Inicializar un nodo en la cuadrícula
        self.x = x  # Coordenada x del nodo
        self.y = y  # Coordenada y del nodo
        self.parent = parent  # Nodo padre para reconstruir el camino
        self.g = 0  # Costo desde el nodo inicial hasta este nodo (camino real)
        self.h = 0  # Heurística: estimación del costo desde este nodo hasta el objetivo (distancia)
        self.f = 0  # Costo total: g + h (costo estimado para el camino más corto)
    
    def __lt__(self, other):
        # Método para comparar nodos basándose en el costo total (f)
        return self.f < other.f

def heuristic(node, goal):
    # Función heurística que calcula la distancia Manhattan
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def a_star(grid, start, goal):
    # Implementación del algoritmo A*
    open_set = []  # Cola de prioridad para nodos abiertos
    closed_set = set()  # Conjunto para nodos cerrados (visitados)
    heapq.heappush(open_set, start)  # Agregar el nodo inicial a la cola de prioridad
    
    while open_set:
        current = heapq.heappop(open_set)  # Extraer el nodo con menor costo total (f)
        
        # Verificar si hemos alcanzado el nodo objetivo
        if current.x == goal.x and current.y == goal.y:
            path = []  # Lista para almacenar el camino encontrado
            while current:
                path.append((current.x, current.y))  # Agregar coordenadas del nodo al camino
                current = current.parent  # Retroceder al nodo padre
            return path[::-1]  # Retornar el camino en orden correcto (inicio a objetivo)
        
        closed_set.add((current.x, current.y))  # Agregar el nodo actual a los nodos cerrados
        
        # Explorar los nodos vecinos (arriba, abajo, izquierda, derecha)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = current.x + dx, current.y + dy  # Nuevas coordenadas
            
            # Verificar si el nuevo nodo está dentro de la cuadrícula y es transitable
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != 1 and (new_x, new_y) not in closed_set:
                new_node = Node(new_x, new_y, current)  # Crear un nuevo nodo vecino
                new_node.g = current.g + 1  # Actualizar el costo g (camino real)
                new_node.h = heuristic(new_node, goal)  # Calcular la heurística h
                new_node.f = new_node.g + new_node.h  # Calcular el costo total f
                heapq.heappush(open_set, new_node)  # Agregar el nuevo nodo a la cola de prioridad
    
    return None  # Retornar None si no se encontró un camino

# Ejemplo de mapa en una cuadrícula (0: camino libre, 1: obstáculo)
grid = [
    [0, 0, 0, 0, 0],  # Fila 0
    [0, 1, 1, 1, 0],  # Fila 1 (obstáculos en el medio)
    [0, 0, 0, 0, 0],  # Fila 2
    [1, 1, 1, 1, 0],  # Fila 3 (obstáculos en el medio)
    [0, 0, 0, 0, 0]   # Fila 4
]

# Punto de inicio y punto objetivo
start = Node(0, 0)  # Nodo de inicio en (0, 0)
goal = Node(4, 4)   # Nodo objetivo en (4, 4)

# Buscar el camino más corto utilizando A*
path = a_star(grid, start, goal)

# Visualizar el camino encontrado
if path:
    print("Camino encontrado:", path)  # Imprimir el camino si se encontró
else:
    print("No se encontró un camino posible.")  # Mensaje si no hay camino
