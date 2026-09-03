"""Clasificador k-NN sobre los embeddings de Hu.

Se implementa con numpy para no agregar dependencias: el conjunto de
entrenamiento es chico y la prediccion es una distancia euclidea contra
unos pocos miles de vectores de 7 dimensiones.

Las 7 componentes de Hu tienen rangos muy distintos entre si, por eso se
estandarizan (media 0, desvio 1) con los estadisticos del entrenamiento
antes de medir distancias.
"""

import numpy as np

from . import config, embeddings

RUTA_MODELO = config.RAIZ / "modelo" / "modelo.npz"


class ClasificadorKNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None
        self.nombres = []
        self.media = None
        self.desvio = None

    # --- entrenamiento -------------------------------------------------
    def entrenar(self, X, y, nombres):
        self.media = X.mean(axis=0)
        self.desvio = X.std(axis=0) + 1e-8
        self.X = (X - self.media) / self.desvio
        self.y = y
        self.nombres = list(nombres)
        return self

    def _normalizar(self, X):
        return (X - self.media) / self.desvio

    # --- prediccion ----------------------------------------------------
    def predecir_vector(self, v):
        """Devuelve (nombre, distancia_al_vecino_mas_cercano, votos).

        distancia sirve como medida de confianza: un embedding lejos de todo
        el entrenamiento corresponde a una forma que el modelo no conoce.
        """
        z = self._normalizar(v.reshape(1, -1))
        distancias = np.linalg.norm(self.X - z, axis=1)
        vecinos = np.argsort(distancias)[: self.k]
        clases, cuentas = np.unique(self.y[vecinos], return_counts=True)
        ganadora = int(clases[np.argmax(cuentas)])
        proporcion = float(np.max(cuentas)) / self.k
        # Distancia al vecino mas cercano de la clase ganadora.
        de_la_clase = distancias[vecinos][self.y[vecinos] == ganadora]
        return self.nombres[ganadora], float(de_la_clase.min()), proporcion

    def predecir_contorno(self, contorno, distancia_maxima=None):
        """Clasifica un contorno; None si esta fuera del umbral de validez."""
        nombre, distancia, proporcion = self.predecir_vector(
            embeddings.vector(contorno)
        )
        if distancia_maxima is not None and distancia > distancia_maxima:
            return None, distancia, proporcion
        return nombre, distancia, proporcion

    def evaluar(self, X, y):
        """Exactitud sobre un conjunto de prueba."""
        aciertos = 0
        for vector_, etiqueta in zip(X, y):
            nombre, _, _ = self.predecir_vector(vector_)
            aciertos += int(self.nombres.index(nombre) == etiqueta)
        return aciertos / len(y)

    def matriz_confusion(self, X, y):
        n = len(self.nombres)
        matriz = np.zeros((n, n), dtype=np.int32)
        for vector_, etiqueta in zip(X, y):
            nombre, _, _ = self.predecir_vector(vector_)
            matriz[etiqueta, self.nombres.index(nombre)] += 1
        return matriz

    # --- persistencia --------------------------------------------------
    def guardar(self, ruta=RUTA_MODELO):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ruta,
            X=self.X,
            y=self.y,
            nombres=np.array(self.nombres),
            media=self.media,
            desvio=self.desvio,
            k=self.k,
        )
        return ruta

    @classmethod
    def cargar(cls, ruta=RUTA_MODELO):
        datos = np.load(ruta, allow_pickle=False)
        modelo = cls(k=int(datos["k"]))
        modelo.X = datos["X"]
        modelo.y = datos["y"]
        modelo.nombres = [str(n) for n in datos["nombres"]]
        modelo.media = datos["media"]
        modelo.desvio = datos["desvio"]
        return modelo


def existe(ruta=RUTA_MODELO):
    return ruta.exists()
