"""Carga y generacion de los contornos de referencia.

Cada clase de objeto se define con una imagen en vision/referencias/.
El nombre del archivo (sin extension) es el nombre de la clase, tal como pide
el enunciado ("usar una imagen de cada uno como referencia, y asignarles un
nombre a cada uno").

Las imagenes que trae el repo son siluetas ideales generadas por este mismo
modulo, para que el proyecto ande recien clonado. Lo que el enunciado pide de
verdad es reemplazarlas por una foto de cada objeto real, con
`python capturar_referencias.py`.
"""

import cv2
import numpy as np

from . import config

LADO_LIENZO = 400


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
    if not directorio.exists():
        return referencias
    for ruta in sorted(directorio.glob("*.png")) + sorted(directorio.glob("*.jpg")):
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"AVISO: no se pudo leer la referencia {ruta}, se ignora.")
            continue
        contorno = contorno_de_imagen(imagen)
        if contorno is not None:
            referencias[ruta.stem] = contorno
    return referencias


# --- Generacion de referencias sinteticas ------------------------------

def _lienzo(lado=LADO_LIENZO):
    return np.full((lado, lado), 255, dtype=np.uint8)


def poligono_regular(lados, radio, rotacion, centro, aspecto=1.0):
    """Vertices de un poligono regular de N lados inscripto en un radio."""
    angulos = np.linspace(0, 2 * np.pi, lados, endpoint=False) + rotacion
    puntos = np.stack(
        [
            centro[0] + radio * aspecto * np.cos(angulos),
            centro[1] + radio * np.sin(angulos),
        ],
        axis=1,
    )
    return puntos.astype(np.int32)


def dibujar_triangulo(lado=LADO_LIENZO):
    """Triangulo EQUILATERO, apuntando hacia arriba.

    Importa que sea equilatero y no un isosceles cualquiera: matchShapes mide
    parecido de forma, no de categoria. Con la referencia isosceles anterior,
    un triangulo equilatero real quedaba a 0.083 de su propia referencia; con
    esta queda a 0.0009, cien veces mas cerca. Eso es lo que permite bajar el
    umbral de validez y que las figuras desconocidas dejen de colarse.
    """
    lienzo = _lienzo(lado)
    centro = (lado / 2, lado / 2)
    puntos = poligono_regular(3, lado * 0.36, -np.pi / 2, centro)
    cv2.fillPoly(lienzo, [puntos], 0)
    return lienzo


def dibujar_cuadrado(lado=LADO_LIENZO):
    lienzo = _lienzo(lado)
    m = int(lado * 0.15)
    cv2.rectangle(lienzo, (m, m), (lado - m, lado - m), 0, -1)
    return lienzo


def dibujar_circulo(lado=LADO_LIENZO):
    lienzo = _lienzo(lado)
    cv2.circle(lienzo, (lado // 2, lado // 2), int(lado * 0.35), 0, -1)
    return lienzo


GENERADORES = {
    "triangulo": dibujar_triangulo,
    "cuadrado": dibujar_cuadrado,
    "circulo": dibujar_circulo,
}


def generar_sinteticas(directorio=None):
    """Crea las imagenes de referencia ideales de las clases conocidas."""
    directorio = directorio or config.DIR_REFERENCIAS
    directorio.mkdir(parents=True, exist_ok=True)
    creadas = []
    for nombre, generador in GENERADORES.items():
        ruta = directorio / f"{nombre}.png"
        cv2.imwrite(str(ruta), generador())
        creadas.append(ruta)
    return creadas
