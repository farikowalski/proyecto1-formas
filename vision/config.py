"""Parametros por defecto y constantes del proyecto.

Ambiente controlado esperado: figuras oscuras sobre fondo claro y homogeneo
(por ejemplo, formas dibujadas con fibron negro sobre una hoja blanca),
iluminacion pareja y camara frontal.
"""

from pathlib import Path

# --- Rutas -------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_REFERENCIAS = RAIZ / "vision" / "referencias"
DIR_CAPTURAS = RAIZ / "capturas"

# --- Clases de objetos -------------------------------------------------
# El nombre del archivo .png en referencias/ define el nombre de la clase.
# El enunciado pide "al menos tres": se pueden agregar mas dejando una imagen
# nueva en referencias/ y sumando aca su nombre, su color y su umbral.
CLASES = ("triangulo", "cuadrado", "circulo")

# Color BGR por clase para anotar la imagen.
COLOR_POR_CLASE = {
    "triangulo": (0, 200, 255),   # naranja
    "cuadrado": (0, 220, 0),      # verde
    "circulo": (255, 180, 0),     # celeste
}
COLOR_DESCONOCIDO = (0, 0, 255)   # rojo
COLOR_DEFECTO = (0, 220, 0)       # verde para clases sin color propio

# --- Umbrales de validez de matchShapes --------------------------------
# El enunciado permite "uno global o uno diferente para cada objeto de
# referencia". Aca se usa uno por clase porque las escalas naturales son muy
# distintas: medido sobre 60 instancias rotadas y escaladas de cada forma, el
# percentil 98 de la distancia a la propia referencia da 0.029 para el
# triangulo pero 0.0018 para el cuadrado y 0.0014 para el circulo. Con un
# unico umbral global de 0.10 (el valor anterior) entraban como "conocidos" el
# 100% de los hexagonos y pentagonos de prueba; con estos, ninguno.
UMBRAL_POR_CLASE = {
    "triangulo": 0.030,
    "cuadrado": 0.005,
    "circulo": 0.005,
}
# Multiplicador global (barra "tolerancia x100") aplicado sobre los umbrales
# por clase: 100 -> los valores de arriba tal cual, 200 -> el doble.
TOLERANCIA_INICIAL = 100
ESCALA_TOLERANCIA = 100.0

# --- Valores iniciales de las barras de desplazamiento ------------------
UMBRAL_INICIAL = 120          # threshold binario [0..255]
# Arranca en manual: el enunciado pide el threshold ajustable con barra y
# ofrece el automatico solo como opcion. Con Otsu prendido la barra "umbral"
# queda inerte, que es justo lo que no conviene mostrar al arrancar.
AUTO_OTSU_INICIAL = 0
KERNEL_INICIAL = 3            # lado del elemento estructural morfologico
AREA_MINIMA_INICIAL = 1500    # area minima del contorno en pixeles
# Distancia k-NN * 100 -> 0.50. Medido sobre 180 figuras reales renderizadas,
# el p99 de la distancia al vecino mas cercano es 0.197 y el maximo 0.209;
# el hexagono mas cercano queda en 0.99 y el pentagono en 1.27. Con 0.50 no
# se rechaza ninguna figura real y se rechazan hexagonos, pentagonos y
# estrellas. El valor anterior (1.50) dejaba pasar hexagonos como circulos.
DISTANCIA_MAXIMA_KNN_INICIAL = 50
FONDO_CLARO_INICIAL = 1       # 1 = objetos oscuros sobre fondo claro

# Maximos de cada barra de desplazamiento.
MAX_UMBRAL = 255
MAX_KERNEL = 21
MAX_AREA = 200                # en unidades de 100 px
MAX_TOLERANCIA = 500          # en centesimas del umbral por clase
MAX_DISTANCIA_KNN = 800
MAX_MARGEN_ROI = 40           # porcentaje recortable por lado

# Los trackbars de distancia trabajan en centesimas para poder usar enteros.
ESCALA_DISTANCIA = 100.0
ESCALA_AREA = 100

# --- Metodos de clasificacion ------------------------------------------
METODO_MATCHSHAPES = "matchshapes"
METODO_EMBEDDING = "embedding"
# Por defecto arranca con matchShapes, que es el metodo que pide el paso 6 del
# enunciado. El k-NN sobre embeddings queda como extra, a un trackbar (o a
# --metodo embedding) de distancia.
METODO_INICIAL = METODO_MATCHSHAPES

# --- Camara ------------------------------------------------------------
INDICE_CAMARA = 0
ANCHO_CAMARA = 1280
ALTO_CAMARA = 720
# Cuadros seguidos que pueden fallar antes de dar la camara por perdida: las
# webcams USB pierden cuadros sueltos con total normalidad.
MAX_FALLOS_CAMARA = 30

# --- Region de interes -------------------------------------------------
# Margen recortado por lado, en porcentaje del cuadro. El enunciado admite las
# dos variantes: la escena es la imagen completa (margenes en 0) o se recorta
# programaticamente el rectangulo de la ROI descartando el resto. Los valores
# se ajustan en vivo con las barras "ROI margen X %" y "ROI margen Y %".
MARGEN_ROI_X_INICIAL = 0
MARGEN_ROI_Y_INICIAL = 0

# --- Ventanas ----------------------------------------------------------
VENTANA_SALIDA = "Deteccion y clasificacion de formas"
VENTANA_BINARIA = "Paso intermedio - binaria"
VENTANA_LIMPIA = "Paso intermedio - morfologia"

# --- Anotacion ---------------------------------------------------------
ALTO_PANEL = 74               # franja superior del panel de estado
