"""Ventana de controles con las barras de desplazamiento.

Cubre los ajustes que el enunciado pide exponer en vivo: el umbral del
threshold (paso 2), el tamano del elemento estructural de la morfologia
(paso 3), el area minima para descartar contornos espureos (paso 5) y el
umbral de distancia maxima de validez (paso 6). Se agregan ademas las dos
barras de la region de interes, para poder recortar la escena sin reiniciar.
"""

import cv2

from . import config

VENTANA_CONTROLES = "Controles"

UMBRAL = "umbral"
AUTO_OTSU = "auto (Otsu)"
KERNEL = "kernel (impar)"
AREA_MINIMA = "area min /100 px"
TOLERANCIA = "tolerancia matchShapes %"
DISTANCIA_KNN = "dist max kNN /100"
FONDO_CLARO = "fondo claro"
METODO = "metodo: 0=matchShapes 1=kNN"
ROI_X = "ROI margen X %"
ROI_Y = "ROI margen Y %"

# Ultimos valores leidos con exito. Si el usuario cierra la ventana de
# controles, getTrackbarPos lanza cv2.error; con esto el bucle sigue andando
# con los ultimos valores en vez de morirse con un traceback.
_ULTIMOS = {}


def _nada(_):
    pass


def crear_controles(metodo_inicial=None, incluir_clasificacion=True):
    """Crea la ventana de controles.

    incluir_clasificacion=False deja solo las barras de segmentacion, que es
    lo unico que necesita capturar_referencias.py.
    """
    metodo_inicial = metodo_inicial or config.METODO_INICIAL
    _ULTIMOS.clear()
    _ULTIMOS["_clasificacion"] = incluir_clasificacion

    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 520, 340)

    cv2.createTrackbar(UMBRAL, VENTANA_CONTROLES,
                       config.UMBRAL_INICIAL, config.MAX_UMBRAL, _nada)
    cv2.createTrackbar(AUTO_OTSU, VENTANA_CONTROLES,
                       config.AUTO_OTSU_INICIAL, 1, _nada)
    cv2.createTrackbar(KERNEL, VENTANA_CONTROLES,
                       config.KERNEL_INICIAL, config.MAX_KERNEL, _nada)
    cv2.createTrackbar(AREA_MINIMA, VENTANA_CONTROLES,
                       config.AREA_MINIMA_INICIAL // config.ESCALA_AREA,
                       config.MAX_AREA, _nada)
    cv2.createTrackbar(FONDO_CLARO, VENTANA_CONTROLES,
                       config.FONDO_CLARO_INICIAL, 1, _nada)
    cv2.createTrackbar(ROI_X, VENTANA_CONTROLES,
                       config.MARGEN_ROI_X_INICIAL, config.MAX_MARGEN_ROI, _nada)
    cv2.createTrackbar(ROI_Y, VENTANA_CONTROLES,
                       config.MARGEN_ROI_Y_INICIAL, config.MAX_MARGEN_ROI, _nada)

    if incluir_clasificacion:
        cv2.createTrackbar(TOLERANCIA, VENTANA_CONTROLES,
                           config.TOLERANCIA_INICIAL, config.MAX_TOLERANCIA, _nada)
        cv2.createTrackbar(DISTANCIA_KNN, VENTANA_CONTROLES,
                           config.DISTANCIA_MAXIMA_KNN_INICIAL,
                           config.MAX_DISTANCIA_KNN, _nada)
        cv2.createTrackbar(
            METODO, VENTANA_CONTROLES,
            0 if metodo_inicial == config.METODO_MATCHSHAPES else 1, 1, _nada,
        )


def existe_ventana():
    """True si la ventana de controles sigue abierta."""
    try:
        return cv2.getWindowProperty(VENTANA_CONTROLES, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _pos(nombre, defecto):
    """Lee una barra tolerando que la ventana ya no exista."""
    try:
        valor = cv2.getTrackbarPos(nombre, VENTANA_CONTROLES)
    except cv2.error:
        valor = -1
    if valor < 0:
        return _ULTIMOS.get(nombre, defecto)
    _ULTIMOS[nombre] = valor
    return valor


def _fijar(nombre, valor):
    """Escribe un valor de vuelta en la barra, para que no mienta en pantalla."""
    try:
        cv2.setTrackbarPos(nombre, VENTANA_CONTROLES, int(valor))
    except cv2.error:
        pass
    _ULTIMOS[nombre] = int(valor)


def sincronizar_umbral(umbral_usado):
    """Refleja en la barra el umbral que Otsu eligio solo.

    Sin esto, con el automatico prendido la barra queda quieta en un numero
    que no es el que se esta aplicando.
    """
    if _pos(AUTO_OTSU, config.AUTO_OTSU_INICIAL):
        _fijar(UMBRAL, umbral_usado)


def leer_controles():
    """Devuelve el diccionario de parametros que consume el pipeline."""
    kernel = _pos(KERNEL, config.KERNEL_INICIAL)
    # El elemento estructural necesita lado impar: se corrige y ademas se
    # escribe de vuelta, asi el slider muestra el valor que realmente se usa.
    if kernel > 0 and kernel % 2 == 0:
        kernel += 1
        _fijar(KERNEL, kernel)

    incluir = _ULTIMOS.get("_clasificacion", True)
    if incluir:
        usa_knn = bool(_pos(METODO, 0))
        metodo = config.METODO_EMBEDDING if usa_knn else config.METODO_MATCHSHAPES
    else:
        metodo = config.METODO_INICIAL

    margen_x = _pos(ROI_X, config.MARGEN_ROI_X_INICIAL) / 100.0
    margen_y = _pos(ROI_Y, config.MARGEN_ROI_Y_INICIAL) / 100.0

    return {
        "umbral": _pos(UMBRAL, config.UMBRAL_INICIAL),
        "auto_otsu": bool(_pos(AUTO_OTSU, config.AUTO_OTSU_INICIAL)),
        "kernel": kernel,
        "area_minima": _pos(AREA_MINIMA, config.AREA_MINIMA_INICIAL // config.ESCALA_AREA)
                       * config.ESCALA_AREA,
        # Se devuelven los dos umbrales por separado: cual se aplica lo decide
        # main.py recien despues de saber que clasificador va a correr de
        # verdad (el pedido puede no estar disponible).
        "tolerancia": _pos(TOLERANCIA, config.TOLERANCIA_INICIAL)
                      / config.ESCALA_TOLERANCIA,
        "distancia_maxima_knn": _pos(DISTANCIA_KNN, config.DISTANCIA_MAXIMA_KNN_INICIAL)
                                / config.ESCALA_DISTANCIA,
        "fondo_claro": bool(_pos(FONDO_CLARO, config.FONDO_CLARO_INICIAL)),
        "metodo": metodo,
        "roi_relativa": (margen_x, margen_y, 1.0 - margen_x, 1.0 - margen_y),
    }


def alternar_metodo():
    """Cambia el metodo desde el teclado (tecla 'm')."""
    _fijar(METODO, 0 if _pos(METODO, 0) else 1)
