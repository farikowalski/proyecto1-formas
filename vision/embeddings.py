"""Embedding de forma: contorno -> vector numerico de dimension fija.

La base son los momentos invariantes de Hu, que no dependen de la traslacion,
la escala ni la rotacion del objeto. Los valores crudos abarcan muchos ordenes
de magnitud, asi que se comprimen con la transformacion logaritmica habitual:

    h'[i] = -sign(h[i]) * log10(|h[i]|)

Cuidado importante: en figuras muy simetricas (un circulo, un cuadrado
perfecto) los momentos de orden alto valen exactamente cero, y ahi el
logaritmo explota o se vuelve indefinido. Medido sobre este proyecto, h3..h7
saltan entre 0 y +-30 segun el ruido del cuadro. Incluso h2 es degenerado:
vale cero exacto en un cuadrado o un circulo ideal. El unico momento estable
en estas tres clases es h1, que ademas las separa por si solo (triangulo
0.714, cuadrado 0.777, circulo 0.797, con desvio del orden de 0.002).

Por eso el embedding toma h1 y lo completa con descriptores geometricos
clasicos, tambien invariantes a rotacion y escala.

Componentes del vector (dimension 6):

    0  log-Hu h1                            tri .71  cua .78  cir .80
    1  circularidad      4*pi*A / P^2       tri .60  cua .79  cir 1.0
    2  convexidad        A / A_envolvente   ~1 en las tres, cae con ruido
    3  llenado del rectangulo minimo        tri .50  cua 1.0  cir .79
    4  llenado del circulo minimo           tri .41  cua .64  cir 1.0
    5  vertices del poligono aproximado /10 tri .3   cua .4   cir >=.7
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
