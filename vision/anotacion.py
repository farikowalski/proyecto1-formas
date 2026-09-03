"""Dibujo de las anotaciones sobre la imagen de salida."""

import cv2

from . import config

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def color_de(nombre):
    if nombre is None:
        return config.COLOR_DESCONOCIDO
    return config.COLOR_POR_CLASE.get(nombre, config.COLOR_DEFECTO)


def _etiqueta(imagen, texto, origen, color):
    x, y = origen
    (ancho, alto), _ = cv2.getTextSize(texto, _FUENTE, 0.6, 2)
    y = max(y, alto + 6)
    cv2.rectangle(imagen, (x, y - alto - 6), (x + ancho + 6, y + 4), color, -1)
    cv2.putText(imagen, texto, (x + 3, y), _FUENTE, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def anotar(imagen, resultados, desplazamiento=(0, 0)):
    """Dibuja contorno, rectangulo y etiqueta de cada objeto clasificado.

    resultados: lista de (contorno, nombre_o_None, distancia).
    desplazamiento: origen de la ROI dentro de la imagen completa.
    """
    dx, dy = desplazamiento
    for contorno, nombre, distancia in resultados:
        color = color_de(nombre)
        desplazado = contorno + (dx, dy)
        cv2.drawContours(imagen, [desplazado], -1, color, 2)

        x, y, w, h = cv2.boundingRect(desplazado)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 1)

        texto = "desconocido" if nombre is None else f"{nombre} ({distancia:.3f})"
        _etiqueta(imagen, texto, (x, y - 8), color)
    return imagen


def panel_estado(imagen, parametros, cantidad, fps=None):
    """Franja superior con los parametros activos y el conteo de objetos."""
    lineas = [
        "umbral={} {}  kernel={}  area_min={}  dist_max={:.2f}".format(
            parametros["umbral_usado"],
            "(Otsu)" if parametros["auto_otsu"] else "(manual)",
            parametros["kernel"],
            parametros["area_minima"],
            parametros["distancia_maxima"],
        ),
        "metodo={}  objetos={}{}".format(
            parametros.get("metodo", "-"),
            cantidad,
            f"  fps={fps:.1f}" if fps else "",
        ),
    ]
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], 52), (0, 0, 0), -1)
    for i, linea in enumerate(lineas):
        cv2.putText(
            imagen, linea, (10, 20 + i * 22), _FUENTE, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return imagen
