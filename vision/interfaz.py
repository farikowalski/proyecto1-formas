"""Ventana de controles con las barras de desplazamiento."""

import cv2

from . import clasificador, config

VENTANA_CONTROLES = "Controles"


def _nada(_):
    pass


def crear_controles(metodo_inicial=None):
    metodo_inicial = metodo_inicial or config.METODO_INICIAL
    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 460, 300)
    cv2.createTrackbar("umbral", VENTANA_CONTROLES, config.UMBRAL_INICIAL, 255, _nada)
    cv2.createTrackbar("auto (Otsu)", VENTANA_CONTROLES, config.AUTO_OTSU_INICIAL, 1, _nada)
    cv2.createTrackbar("kernel", VENTANA_CONTROLES, config.KERNEL_INICIAL, 21, _nada)
    cv2.createTrackbar("area min /100", VENTANA_CONTROLES,
                       config.AREA_MINIMA_INICIAL // 100, 200, _nada)
    cv2.createTrackbar("dist max matchShapes /100", VENTANA_CONTROLES,
                       config.DISTANCIA_MAXIMA_MS_INICIAL, 200, _nada)
    cv2.createTrackbar("dist max kNN /100", VENTANA_CONTROLES,
                       config.DISTANCIA_MAXIMA_KNN_INICIAL, 800, _nada)
    cv2.createTrackbar("fondo claro", VENTANA_CONTROLES, config.FONDO_CLARO_INICIAL, 1, _nada)
    cv2.createTrackbar(
        "metodo: 0=kNN 1=matchShapes", VENTANA_CONTROLES,
        0 if metodo_inicial == clasificador.METODO_EMBEDDING else 1, 1, _nada,
    )


def leer_controles():
    kernel = cv2.getTrackbarPos("kernel", VENTANA_CONTROLES)
    # El elemento estructural necesita lado impar.
    if kernel > 0 and kernel % 2 == 0:
        kernel += 1

    usa_matchshapes = bool(
        cv2.getTrackbarPos("metodo: 0=kNN 1=matchShapes", VENTANA_CONTROLES)
    )
    metodo = (
        clasificador.METODO_MATCHSHAPES if usa_matchshapes
        else clasificador.METODO_EMBEDDING
    )
    clave = "dist max matchShapes /100" if usa_matchshapes else "dist max kNN /100"

    return {
        "umbral": cv2.getTrackbarPos("umbral", VENTANA_CONTROLES),
        "auto_otsu": bool(cv2.getTrackbarPos("auto (Otsu)", VENTANA_CONTROLES)),
        "kernel": kernel,
        "area_minima": cv2.getTrackbarPos("area min /100", VENTANA_CONTROLES) * 100,
        "distancia_maxima": (
            cv2.getTrackbarPos(clave, VENTANA_CONTROLES) / config.ESCALA_DISTANCIA
        ),
        "fondo_claro": bool(cv2.getTrackbarPos("fondo claro", VENTANA_CONTROLES)),
        "metodo": metodo,
    }
