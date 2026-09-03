"""Clasificacion de contornos con dos metodos intercambiables.

1. matchShapes: compara el contorno contra el contorno de referencia de cada
   clase. Es el metodo que pide el paso 6 del enunciado, y el que corre por
   defecto. Por dentro OpenCV usa los mismos momentos de Hu en escala
   logaritmica.

2. embedding + k-NN: convierte el contorno en un vector de Hu (ver
   embeddings.py) y lo clasifica con un modelo entrenado sobre muchas
   variantes de cada forma. Generaliza mejor que una unica referencia.

Ambos devuelven (nombre, distancia); nombre es None cuando ningun candidato
queda dentro de su umbral de validez, es decir cuando la forma es desconocida.
"""

import cv2

from . import config, embeddings

METODO_MATCHSHAPES = config.METODO_MATCHSHAPES
METODO_EMBEDDING = config.METODO_EMBEDDING


class ClasificadorMatchShapes:
    """Compara cada contorno con TODOS los objetos de referencia.

    El enunciado permite un umbral de validez "global o uno diferente para
    cada objeto de referencia". Aca se usa uno por clase (config.UMBRAL_POR_CLASE)
    porque las escalas naturales son muy distintas entre formas, y se ofrece
    una tolerancia global que los multiplica a todos para poder aflojarlos o
    apretarlos en vivo con una sola barra.
    """

    nombre_metodo = METODO_MATCHSHAPES

    def __init__(self, referencias, metodo=cv2.CONTOURS_MATCH_I1, umbrales=None):
        self.referencias = referencias
        self.metodo = metodo
        self.umbrales = dict(umbrales or config.UMBRAL_POR_CLASE)

    @property
    def clases(self):
        return sorted(self.referencias)

    def umbral_de(self, nombre, tolerancia=1.0):
        """Umbral de validez de una clase, escalado por la tolerancia global."""
        base = self.umbrales.get(nombre)
        if base is None:
            # Clase sin umbral propio: se usa el mas permisivo de los conocidos.
            base = max(self.umbrales.values()) if self.umbrales else 0.03
        return base * tolerancia

    def distancias(self, contorno):
        return {
            nombre: cv2.matchShapes(contorno, referencia, self.metodo, 0.0)
            for nombre, referencia in self.referencias.items()
        }

    def clasificar(self, contorno, tolerancia=1.0):
        """Devuelve (nombre, distancia); nombre None si no hay candidato valido."""
        if not self.referencias:
            return None, float("inf")

        medidas = self.distancias(contorno)
        # Solo son candidatos los que estan dentro de SU propio umbral...
        validos = {
            nombre: distancia
            for nombre, distancia in medidas.items()
            if distancia <= self.umbral_de(nombre, tolerancia)
        }
        if not validos:
            # ...y si no queda ninguno, la forma es desconocida. Se informa la
            # menor distancia igual, que es el dato util para ajustar el umbral.
            return None, min(medidas.values())
        # ...y entre los validos gana el de menor distancia.
        nombre, distancia = min(validos.items(), key=lambda par: par[1])
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


def clasificar_todos(contornos, clasificador, umbral):
    """Aplica el clasificador a cada contorno detectado en la escena.

    `umbral` es la tolerancia global para matchShapes y la distancia maxima
    para el k-NN: cada clasificador interpreta el numero a su manera.
    """
    return [
        (contorno,) + clasificador.clasificar(contorno, umbral)
        for contorno in contornos
    ]
