"""Clasificacion de contornos con dos metodos intercambiables.

1. matchShapes: compara el contorno contra el contorno de referencia de cada
   clase. Es el metodo que pide el enunciado del proyecto. Por dentro OpenCV
   usa los mismos momentos de Hu en escala logaritmica.

2. embedding + k-NN: convierte el contorno en un vector de Hu (ver
   embeddings.py) y lo clasifica con un modelo entrenado sobre muchas
   variantes de cada forma. Generaliza mejor que una unica referencia.

Ambos devuelven (nombre, distancia); nombre es None cuando la distancia
supera el umbral de validez, es decir cuando la forma es desconocida.
"""

import cv2

from . import embeddings

METODO_MATCHSHAPES = "matchshapes"
METODO_EMBEDDING = "embedding"


class ClasificadorMatchShapes:
    nombre_metodo = METODO_MATCHSHAPES

    def __init__(self, referencias, metodo=cv2.CONTOURS_MATCH_I1):
        self.referencias = referencias
        self.metodo = metodo

    @property
    def clases(self):
        return sorted(self.referencias)

    def distancias(self, contorno):
        return {
            nombre: cv2.matchShapes(contorno, referencia, self.metodo, 0.0)
            for nombre, referencia in self.referencias.items()
        }

    def clasificar(self, contorno, distancia_maxima):
        if not self.referencias:
            return None, float("inf")
        medidas = self.distancias(contorno)
        nombre, distancia = min(medidas.items(), key=lambda par: par[1])
        if distancia > distancia_maxima:
            return None, distancia
        return nombre, distancia


class ClasificadorEmbedding:
    nombre_metodo = METODO_EMBEDDING

    def __init__(self, modelo):
        self.modelo = modelo

    @property
    def clases(self):
        return list(self.modelo.nombres)

    def vector(self, contorno):
        return embeddings.vector(contorno)

    def clasificar(self, contorno, distancia_maxima):
        nombre, distancia, _ = self.modelo.predecir_contorno(
            contorno, distancia_maxima
        )
        return nombre, distancia


def clasificar_todos(contornos, clasificador, distancia_maxima):
    """Aplica el clasificador a cada contorno detectado en la escena."""
    return [
        (contorno,) + clasificador.clasificar(contorno, distancia_maxima)
        for contorno in contornos
    ]
