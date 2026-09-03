"""Parametros por defecto y constantes del proyecto.

Ambiente controlado esperado: figuras oscuras sobre fondo claro y homogeneo
(por ejemplo, formas dibujadas con fibron negro sobre una hoja blanca),
iluminacion pareja y camara frontal.
"""

from pathlib import Path

# --- Rutas -------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_REFERENCIAS = RAIZ / "vision" / "referencias"

# --- Clases de objetos -------------------------------------------------
# El nombre del archivo .png en referencias/ define el nombre de la clase.
CLASES = ("triangulo", "cuadrado", "circulo")

# Color BGR por clase para anotar la imagen.
COLOR_POR_CLASE = {
    "triangulo": (0, 200, 255),   # naranja
    "cuadrado": (0, 220, 0),      # verde
    "circulo": (255, 180, 0),     # celeste
}
COLOR_DESCONOCIDO = (0, 0, 255)   # rojo
COLOR_DEFECTO = (0, 220, 0)       # verde para clases sin color propio

# --- Valores iniciales de las barras de desplazamiento ------------------
UMBRAL_INICIAL = 120          # threshold binario [0..255]
AUTO_OTSU_INICIAL = 1         # 1 = umbral automatico (Otsu) activado
KERNEL_INICIAL = 3            # lado del elemento estructural morfologico
AREA_MINIMA_INICIAL = 1500    # area minima del contorno en pixeles
DISTANCIA_MAXIMA_MS_INICIAL = 10   # matchShapes * 100 -> 0.10
DISTANCIA_MAXIMA_KNN_INICIAL = 150 # distancia k-NN * 100 -> 1.50
FONDO_CLARO_INICIAL = 1       # 1 = objetos oscuros sobre fondo claro

# Los trackbars de distancia trabajan en centesimas para poder usar enteros.
ESCALA_DISTANCIA = 100.0

# Metodo de clasificacion por defecto: "embedding" (k-NN sobre Hu) o
# "matchshapes" (comparacion directa contra el contorno de referencia).
METODO_INICIAL = "embedding"

# --- Camara ------------------------------------------------------------
INDICE_CAMARA = 0
ANCHO_CAMARA = 1280
ALTO_CAMARA = 720

# Region de interes como fraccion del cuadro (x1, y1, x2, y2).
# (0, 0, 1, 1) usa la imagen completa.
ROI_RELATIVA = (0.0, 0.0, 1.0, 1.0)
