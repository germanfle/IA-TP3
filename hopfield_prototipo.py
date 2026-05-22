# ================================================================================
# TP3 - Prototipo Red de Hopfield para identificacion del Aro C
# Problema: Linea de montaje industrial con grilla 10x10 (100 neuronas)
# Materia: INF404 Inteligencia Artificial – Licenciatura en Informática - Siglo 21
# ================================================================================

import numpy as np

class RedHopfield:
    #Red de Hopfield: memoria asociativa de una sola capa completamente conectada
    def __init__(self, n_neuronas):
        # Matriz de pesos sinapticos NxN, inicializada en cero
        self.W = np.zeros((n_neuronas, n_neuronas))

    def entrenar(self, patrones):
        #APRENDIZAJE: almacena patrones con la regla de Hebb
        #W = sum(xi * xi^T) para cada patron xi, con diagonal nula
        for p in patrones:
            self.W += np.outer(p, p)   #Producto externo: refuerza conexiones co-activas
        np.fill_diagonal(self.W, 0)    #Sin auto-conexiones (wii = 0)
        self.W /= len(patrones)        #Normalizacion por numero de patrones almacenados

    def recuperar(self, patron, pasos=10):
        #RECUPERACION: actualizacion asincronica hasta estabilizar el estado
        #Regla de actualizacion: si = sgn(sum_j wij * sj)
        estado = patron.copy()
        for _ in range(pasos):
            for i in np.random.permutation(len(estado)):  #Orden aleatorio = asincrono
                estado[i] = 1 if np.dot(self.W[i], estado) > 0 else -1
        return estado

    def energia(self, patron):
        #Funcion de energia de Lyapunov: E = -0.5 * s^T * W * s
        #Decrece en cada actualizacion, garantizando convergencia a un minimo local
        return -0.5 * np.dot(patron.T, np.dot(self.W, patron))

#Funciones auxiliares para la linea de montaje
def crear_aro(cx=5, cy=5):
    #Genera el patron binario del aro C en una grilla 10x10
    #Pixeles con distancia al centro entre 2 y 3 valen +1 (aro); resto -1 (fondo)
    grid = np.full((10, 10), -1)
    for i in range(10):
        for j in range(10):
            if 2 <= np.sqrt((i - cx)**2 + (j - cy)**2) <= 3:
                grid[i, j] = 1
    return grid.flatten()  #Vector de 100 elementos para alimentar la red

def aplicar_ruido(patron, nivel=0.2):
    #Simula degradacion de imagen de camara invirtiendo un porcentaje de pixeles
    ruido = patron.copy()
    indices = np.random.choice(len(ruido), int(nivel * len(ruido)), replace=False)
    ruido[indices] *= -1   #flip: +1 pasa a -1 o viceversa
    return ruido

def calcular_centro(imagen):
    #Estima la posicion (X, Y) del aro a partir del centroide de los pixeles activos
    grid = imagen.reshape(10, 10)
    y, x = np.where(grid == 1)
    return (int(np.mean(x)), int(np.mean(y))) if len(x) > 0 else (None, None)

#Ejecucion del prototipo
if __name__ == "__main__":

    # 1. Crea red de 100 neuronas (grilla 10x10)
    red = RedHopfield(100)
    # 2. APRENDIZAJE: almacena el aro en posicion correcta y dos desplazamientos posibles
    patrones_entrenamiento = [crear_aro(5, 5), crear_aro(4, 5), crear_aro(6, 6)]
    red.entrenar(patrones_entrenamiento)
    # 3. Simula captura de camara: imagen del aro con 25% de ruido
    np.random.seed(42)                          #Semilla fija para reproducibilidad
    aro_ideal     = patrones_entrenamiento[0]   #Referencia: aro centrado en (5,5)
    aro_capturado = aplicar_ruido(aro_ideal, nivel=0.25)
    # 4. RECUPERACION: la red reconstruye el patron a partir de la imagen ruidosa
    aro_limpio = red.recuperar(aro_capturado)
    # 5. Estima coordenadas del aro para el sistema de posicionamiento del robot
    cx, cy = calcular_centro(aro_limpio)

    print(f"Energia de la imagen con ruido: {red.energia(aro_capturado):.2f}")
    print(f"Energia tras estabilizacion:    {red.energia(aro_limpio):.2f}")
    print(f"Posicion detectada para el robot (X, Y): ({cx}, {cy})")