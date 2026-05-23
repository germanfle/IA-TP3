# ================================================================================
# TP3 - Prototipo Red de Hopfield para identificacion del Aro C
# Problema: Linea de montaje industrial con grilla 10x10 (100 neuronas)
# Materia: INF404 Inteligencia Artificial – Licenciatura en Informática - Siglo 21
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

class RedHopfield:
    """Red de Hopfield: memoria asociativa de una sola capa completamente conectada."""

    def __init__(self, n_neuronas):
        self.W = np.zeros((n_neuronas, n_neuronas))

    def entrenar_hebb(self, patrones):
        """Fase de APRENDIZAJE con regla de Hebb (1949).
        W = sum(xi * xi^T) para cada patron xi, con diagonal nula.
        Requiere patrones aproximadamente ortogonales entre si."""
        self.W = np.zeros_like(self.W)
        for p in patrones:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.W /= len(patrones)

    def entrenar_pseudoinversa(self, patrones):
        """Fase de APRENDIZAJE con regla de la pseudoinversa (Personnaz, 1986).
        W = X * X^+ donde X^+ es la pseudoinversa de Moore-Penrose de X.
        Supera la necesidad de ortogonalidad; soporta patrones correlacionados."""
        X = np.column_stack(patrones)
        self.W = X @ np.linalg.pinv(X)
        np.fill_diagonal(self.W, 0)

    def recuperar(self, patron, pasos=10):
        """Fase de RECUPERACION: actualizacion asincronica hasta estabilizar.
        Regla: si = sgn(sum_j wij * sj). Orden de actualizacion aleatorio."""
        estado = patron.copy()
        for _ in range(pasos):
            for i in np.random.permutation(len(estado)):
                estado[i] = 1 if np.dot(self.W[i], estado) > 0 else -1
        return estado

    def energia(self, patron):
        """Funcion de energia de Lyapunov: E = -0.5 * s^T * W * s"""
        return -0.5 * np.dot(patron.T, np.dot(self.W, patron))


# --- Funciones auxiliares ---

def crear_aro(x=5, y=5):
    """Genera el patron binario del aro C en una grilla 10x10.
    x = columna del centro, y = fila del centro.
    Pixeles con distancia al centro entre 2.5 y 4.0 valen +1.
    Para evitar efecto de borde usar centros en rango 4-5 (radio maximo = 4)."""
    grid = np.full((10, 10), -1)
    for fila in range(10):
        for col in range(10):
            if 2.5 <= np.sqrt((fila - y)**2 + (col - x)**2) <= 4.0:
                grid[fila, col] = 1
    return grid.flatten()

def crear_escuadra():
    """Genera la escuadra de referencia en el angulo inferior izquierdo.
    Posicion INALTERABLE — permite medir el desplazamiento relativo del aro.
    Forma L: barra vertical (col 0, filas 7-9) + barra horizontal (fila 9, cols 0-2)."""
    grid = np.full((10, 10), -1)
    for fila in range(7, 10):
        grid[fila, 0] = 1
    for col in range(0, 3):
        grid[9, col] = 1
    return grid.flatten()

def combinar_patron(aro, escuadra):
    """Superpone el aro y la escuadra en un unico patron binario."""
    return np.where((aro == 1) | (escuadra == 1), 1, -1)

def aplicar_ruido(patron, nivel=0.2):
    """Simula degradacion de imagen invirtiendo un porcentaje de pixeles."""
    ruido = patron.copy()
    indices = np.random.choice(len(ruido), int(nivel * len(ruido)), replace=False)
    ruido[indices] *= -1
    return ruido

_MASCARA_ESCUADRA = crear_escuadra() == 1

def calcular_centro(imagen):
    """Estima la posicion (X, Y) del aro excluyendo los pixeles de la escuadra.
    Usa redondeo para evitar el sesgo de truncamiento en posiciones no centrales."""
    solo_aro = imagen.copy()
    solo_aro[_MASCARA_ESCUADRA] = -1
    grid = solo_aro.reshape(10, 10)
    filas, cols = np.where(grid == 1)
    if len(cols) == 0:
        return (None, None)
    return (int(round(np.mean(cols))), int(round(np.mean(filas))))


def mostrar_resultado(red_h, red_p, patrones, nombres, nivel=0.20):
    """Compara Hebb vs. Pseudoinversa para los tres patrones almacenados.
    Grilla 3 filas x 4 columnas:
      Col 1 — patron original (aro + escuadra)
      Col 2 — imagen capturada con ruido
      Col 3 — recuperado con Hebb (1949)
      Col 4 — recuperado con Pseudoinversa (Personnaz 1986)"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    titulos_col = ["Original",
                   f"Con ruido\n({int(nivel*100)}%)",
                   "Recuperado\n(Hebb)",
                   "Recuperado\n(Pseudoinversa)"]
    for c, t in enumerate(titulos_col):
        axes[0][c].set_title(t, fontsize=9, fontweight="bold")

    for fila, (patron, nombre) in enumerate(zip(patrones, nombres)):
        np.random.seed(42)
        ruidoso = aplicar_ruido(patron, nivel=nivel)
        rec_h   = red_h.recuperar(ruidoso)
        rec_p   = red_p.recuperar(ruidoso)
        cx_h, cy_h = calcular_centro(rec_h)
        cx_p, cy_p = calcular_centro(rec_p)

        imagenes = [patron, ruidoso, rec_h,          rec_p]
        centros  = [None,   None,    (cx_h, cy_h),   (cx_p, cy_p)]

        for col, (img, ctr) in enumerate(zip(imagenes, centros)):
            ax = axes[fila][col]
            ax.imshow(img.reshape(10, 10), cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(nombre, fontsize=8, rotation=90, labelpad=4)
            if ctr and ctr[0] is not None:
                ax.plot(ctr[0], ctr[1], "r+", markersize=12,
                        markeredgewidth=2, label=f"({ctr[0]},{ctr[1]})")
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Red de Hopfield 10x10 — Hebb vs. Pseudoinversa\n"
        "Escuadra (ang. inf. izq.): referencia fija | Cruz roja: posicion del aro",
        fontsize=10
    )
    plt.tight_layout()
    directorio = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(directorio, "resultado_aro.png")
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Imagen guardada en: {ruta}")


# --- Ejecucion del prototipo ---

if __name__ == "__main__":

    # 1. Red de 100 neuronas (grilla 10x10)
    red_h = RedHopfield(100)
    red_p = RedHopfield(100)

    # 2. Escuadra de referencia fija
    escuadra = crear_escuadra()

    # 3. Tres patrones de entrenamiento: aro en posiciones dentro del rango seguro
    #    (radio max = 4.0, centros en cols/filas 4-5 evitan efecto de borde)
    #    Convencion: crear_aro(x=columna, y=fila)
    patrones = [
        combinar_patron(crear_aro(x=5, y=5), escuadra),  # centrado
        combinar_patron(crear_aro(x=4, y=5), escuadra),  # desplazado -1 en X
        combinar_patron(crear_aro(x=5, y=4), escuadra),  # desplazado -1 en Y
    ]
    nombres = ["Aro centrado  (X=5,Y=5)",
               "Desplaz. X    (X=4,Y=5)",
               "Desplaz. Y    (X=5,Y=4)"]

    # 4. Entrenamiento con ambos metodos
    red_h.entrenar_hebb(patrones)
    red_p.entrenar_pseudoinversa(patrones)

    # 5. Tabla de resultados (20% de ruido)
    print("=" * 76)
    print(f"{'Patron':<26} | {'Hebb':^22} | {'Pseudoinversa':^22}")
    print("=" * 76)
    esperados = [(5, 5), (4, 5), (5, 4)]
    for patron, nombre, esp in zip(patrones, nombres, esperados):
        np.random.seed(42)
        ruidoso = aplicar_ruido(patron, nivel=0.20)
        ch = calcular_centro(red_h.recuperar(ruidoso))
        cp = calcular_centro(red_p.recuperar(ruidoso))
        ok_h = "OK" if ch == esp else "FALLO"
        ok_p = "OK" if cp == esp else "FALLO"
        print(f"{nombre:<26} | pos={ch}  Eh={red_h.energia(red_h.recuperar(patron)):7.1f} {ok_h:5}"
              f" | pos={cp}  Ep={red_p.energia(red_p.recuperar(patron)):7.1f} {ok_p:5}")
    print("=" * 76)

    # 6. Visualizacion: Hebb vs. Pseudoinversa para los tres patrones
    mostrar_resultado(red_h, red_p, patrones, nombres, nivel=0.20)
