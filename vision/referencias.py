"""Carga y generacion de los contornos de referencia.

Cada clase de objeto se define con una imagen en vision/referencias/.
El nombre del archivo (sin extension) es el nombre de la clase.
"""

import cv2
import numpy as np

from . import config


def _contorno_mayor(binaria):
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def contorno_de_imagen(imagen):
    """Extrae el contorno principal de una imagen de referencia."""
    if imagen.ndim == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen
    _, binaria = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return _contorno_mayor(binaria)


def cargar(directorio=None):
    """Devuelve {nombre_clase: contorno} leyendo las imagenes de referencia."""
    directorio = directorio or config.DIR_REFERENCIAS
    referencias = {}
    for ruta in sorted(directorio.glob("*.png")) + sorted(directorio.glob("*.jpg")):
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            continue
        contorno = contorno_de_imagen(imagen)
        if contorno is not None:
            referencias[ruta.stem] = contorno
    return referencias


# --- Generacion de referencias sinteticas ------------------------------

def _lienzo(lado=400):
    return np.full((lado, lado), 255, dtype=np.uint8)


def dibujar_triangulo(lado=400):
    lienzo = _lienzo(lado)
    m = int(lado * 0.12)
    puntos = np.array(
        [[lado // 2, m], [m, lado - m], [lado - m, lado - m]], dtype=np.int32
    )
    cv2.fillPoly(lienzo, [puntos], 0)
    return lienzo


def dibujar_cuadrado(lado=400):
    lienzo = _lienzo(lado)
    m = int(lado * 0.15)
    cv2.rectangle(lienzo, (m, m), (lado - m, lado - m), 0, -1)
    return lienzo


def dibujar_circulo(lado=400):
    lienzo = _lienzo(lado)
    cv2.circle(lienzo, (lado // 2, lado // 2), int(lado * 0.35), 0, -1)
    return lienzo


GENERADORES = {
    "triangulo": dibujar_triangulo,
    "cuadrado": dibujar_cuadrado,
    "circulo": dibujar_circulo,
}


def generar_sinteticas(directorio=None):
    """Crea las imagenes de referencia ideales de las tres clases."""
    directorio = directorio or config.DIR_REFERENCIAS
    directorio.mkdir(parents=True, exist_ok=True)
    creadas = []
    for nombre, generador in GENERADORES.items():
        ruta = directorio / f"{nombre}.png"
        cv2.imwrite(str(ruta), generador())
        creadas.append(ruta)
    return creadas
