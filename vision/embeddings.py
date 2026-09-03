"""Embedding de forma: contorno -> vector numerico de dimension fija.

La base son los momentos invariantes de Hu, que no dependen de la traslacion,
la escala ni la rotacion del objeto. Los valores crudos abarcan muchos ordenes
de magnitud, asi que se comprimen con la transformacion logaritmica habitual:

    h'[i] = -sign(h[i]) * log10(|h[i]|)

Cuidado importante: en figuras muy simetricas (un circulo, un cuadrado
perfecto) los momentos de orden alto valen exactamente cero, y ahi el
logaritmo explota o se vuelve indefinido. Medido sobre 200 circulos con el
ruido de camara del generador, h4 recorre todo el rango 0..12 (desvio 5.62) y
h6 salta de 0 a 11.74, mientras un circulo ideal da cero exacto en h2..h7.
El tope de esos saltos lo pone _TOPE_LOG, mas abajo. El unico momento estable
en estas tres clases es h1, que ademas las separa por si solo (triangulo
0.715, cuadrado 0.778, circulo 0.797, con desvio del orden de 0.001).

Por eso el embedding toma h1 y lo completa con descriptores geometricos
clasicos, tambien invariantes a rotacion y escala.

Componentes del vector (dimension 6):

    0  log-Hu h1                            tri .715 cua .778 cir .797
    1  circularidad      4*pi*A / P^2       tri .55  cua .71  cir .88
    2  convexidad        A / A_envolvente   tri .98  cua .99  cir .99
    3  llenado del rectangulo minimo        tri .51  cua .96  cir .79
    4  llenado del circulo minimo           tri .43  cua .63  cir .94
    5  vertices del poligono aproximado /10 tri .30  cua .40  cir .80

Los valores son las medias medidas sobre el dataset; los limites teoricos del
continuo (circularidad 1.0 en un circulo) no se alcanzan porque el contorno
esta discretizado en pixeles.
"""

import cv2
import numpy as np

DIMENSION = 6
_MINIMO_HU = 1e-12   # por debajo de esto el momento es ruido numerico
_TOPE_LOG = 12.0     # cota del valor logaritmico, para no propagar infinitos

NOMBRES_COMPONENTES = (
    "log_hu1",
    "circularidad",
    "convexidad",
    "llenado_rectangulo",
    "llenado_circulo",
    "vertices/10",
)


def log_hu(contorno):
    """Los 7 momentos de Hu en escala logaritmica, acotados.

    Se expone completo con fines didacticos y para inspeccion; el embedding
    usado por el clasificador toma solo la primera componente.
    """
    hu = cv2.HuMoments(cv2.moments(contorno)).flatten()
    magnitud = np.abs(hu)
    # Los momentos indistinguibles de cero no aportan informacion: valen 0.
    valores = np.where(
        magnitud < _MINIMO_HU,
        0.0,
        -np.sign(hu) * np.log10(np.maximum(magnitud, _MINIMO_HU)),
    )
    return np.clip(valores, -_TOPE_LOG, _TOPE_LOG)


def descriptores_geometricos(contorno):
    """Cinco relaciones adimensionales del contorno, invariantes a pose."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    if area <= 0 or perimetro <= 0:
        return np.zeros(5, dtype=np.float64)

    circularidad = 4.0 * np.pi * area / (perimetro ** 2)

    envolvente = cv2.convexHull(contorno)
    area_envolvente = cv2.contourArea(envolvente)
    convexidad = area / area_envolvente if area_envolvente > 0 else 0.0

    (_, _), (ancho, alto), _ = cv2.minAreaRect(contorno)
    area_rectangulo = ancho * alto
    llenado_rectangulo = area / area_rectangulo if area_rectangulo > 0 else 0.0

    _, radio = cv2.minEnclosingCircle(contorno)
    area_circulo = np.pi * radio ** 2
    llenado_circulo = area / area_circulo if area_circulo > 0 else 0.0

    # approxPolyDP con tolerancia proporcional al perimetro cuenta los lados.
    aproximado = cv2.approxPolyDP(contorno, 0.03 * perimetro, True)
    vertices = min(len(aproximado), 10) / 10.0

    return np.array(
        [circularidad, convexidad, llenado_rectangulo, llenado_circulo, vertices],
        dtype=np.float64,
    )


def vector(contorno):
    """Embedding completo del contorno."""
    hu = log_hu(contorno)[:1]
    return np.concatenate([hu, descriptores_geometricos(contorno)]).astype(np.float32)


def matriz(contornos):
    """Apila los embeddings de varios contornos en una matriz (n, 6)."""
    if not len(contornos):
        return np.empty((0, DIMENSION), dtype=np.float32)
    return np.vstack([vector(c) for c in contornos]).astype(np.float32)
