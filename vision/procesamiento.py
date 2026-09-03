"""Pipeline de procesamiento: gris -> threshold -> morfologia -> contornos."""

import cv2
import numpy as np


def recortar_roi(imagen, roi_relativa):
    """Recorta la region de interes expresada en fracciones del cuadro."""
    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = roi_relativa
    px1, py1 = int(x1 * ancho), int(y1 * alto)
    px2, py2 = int(x2 * ancho), int(y2 * alto)
    px2 = max(px2, px1 + 1)
    py2 = max(py2, py1 + 1)
    return imagen[py1:py2, px1:px2], (px1, py1)


def a_gris(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def binarizar(gris, umbral, auto_otsu, fondo_claro):
    """Devuelve (binaria, umbral_usado).

    La binaria siempre queda con el objeto en blanco (255) y el fondo en negro,
    que es lo que findContours espera.
    """
    # Un desenfoque suave reduce el ruido del sensor antes de umbralizar.
    suave = cv2.GaussianBlur(gris, (5, 5), 0)

    tipo = cv2.THRESH_BINARY_INV if fondo_claro else cv2.THRESH_BINARY
    if auto_otsu:
        umbral_usado, binaria = cv2.threshold(
            suave, 0, 255, tipo | cv2.THRESH_OTSU
        )
    else:
        umbral_usado, binaria = cv2.threshold(suave, umbral, 255, tipo)
    return binaria, umbral_usado


def limpiar(binaria, lado_kernel):
    """Operaciones morfologicas para eliminar ruido y cerrar huecos."""
    if lado_kernel < 1:
        return binaria
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (lado_kernel, lado_kernel)
    )
    # Apertura: borra puntos sueltos. Cierre: rellena cortes del contorno.
    sin_ruido = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(sin_ruido, cv2.MORPH_CLOSE, kernel)


def buscar_contornos(binaria, area_minima):
    """Contornos externos que superan el area minima y no tocan el borde."""
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    alto, ancho = binaria.shape[:2]
    utiles = []
    for contorno in contornos:
        if cv2.contourArea(contorno) < area_minima:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        # Un objeto cortado por el borde tiene un contorno falso: se descarta.
        if x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1:
            continue
        utiles.append(contorno)
    return utiles


def procesar(imagen, parametros):
    """Ejecuta el pipeline completo y devuelve los pasos intermedios."""
    gris = a_gris(imagen)
    binaria, umbral_usado = binarizar(
        gris,
        parametros["umbral"],
        parametros["auto_otsu"],
        parametros["fondo_claro"],
    )
    limpia = limpiar(binaria, parametros["kernel"])
    contornos = buscar_contornos(limpia, parametros["area_minima"])
    return {
        "gris": gris,
        "binaria": binaria,
        "limpia": limpia,
        "contornos": contornos,
        "umbral_usado": umbral_usado,
    }
