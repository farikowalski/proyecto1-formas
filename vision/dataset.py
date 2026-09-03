"""Generacion del conjunto de entrenamiento de embeddings.

Cada muestra es una imagen sintetica de una figura (o una version aumentada
de una foto de referencia del alumno) de la que se extrae el contorno y se
calcula su embedding de Hu. Las variaciones aleatorias de rotacion, escala,
proporcion y ruido hacen que el clasificador aprenda la forma y no una pose
particular.
"""

import cv2
import numpy as np

from . import config, embeddings

LADO = 400

# Clases que se pueden dibujar como poligono regular. El resto se genera
# aumentando su imagen de referencia, asi agregar una cuarta clase no
# obliga a escribir un generador nuevo.
LADOS_POR_CLASE = {"triangulo": 3, "cuadrado": 4}


def _lienzo():
    return np.full((LADO, LADO), 255, dtype=np.uint8)


def _poligono_regular(lados, radio, rotacion):
    angulos = np.linspace(0, 2 * np.pi, lados, endpoint=False) + rotacion
    centro = LADO / 2
    puntos = np.stack(
        [centro + radio * np.cos(angulos), centro + radio * np.sin(angulos)], axis=1
    )
    return puntos.astype(np.int32)


def _dibujar(clase, rng):
    """Dibuja una instancia aleatoria de la clase pedida."""
    lienzo = _lienzo()
    radio = rng.uniform(0.22, 0.42) * LADO
    rotacion = rng.uniform(0, 2 * np.pi)

    lados = LADOS_POR_CLASE.get(clase)
    if lados:
        cv2.fillPoly(lienzo, [_poligono_regular(lados, radio, rotacion)], 0)
    elif clase == "circulo":
        centro = (LADO // 2, LADO // 2)
        cv2.circle(lienzo, centro, int(radio), 0, -1)
    else:
        # Clase sin generador sintetico propio: sus muestras salen de
        # aumentar su imagen de referencia (ver construir()).
        return None
    return lienzo


def _deformar(imagen, rng):
    """Perspectiva suave, desplazamiento y ruido, como los de una webcam."""
    # Estiramiento leve para simular que la camara no esta perfectamente frontal.
    escala_x = rng.uniform(0.92, 1.08)
    escala_y = rng.uniform(0.92, 1.08)
    dx = rng.uniform(-0.06, 0.06) * LADO
    dy = rng.uniform(-0.06, 0.06) * LADO
    centro = LADO / 2
    afin = np.array(
        [
            [escala_x, 0, centro * (1 - escala_x) + dx],
            [0, escala_y, centro * (1 - escala_y) + dy],
        ],
        dtype=np.float32,
    )
    salida = cv2.warpAffine(
        imagen, afin, (LADO, LADO), borderValue=255,
        flags=cv2.INTER_LINEAR,
    )
    if rng.random() < 0.5:
        salida = cv2.GaussianBlur(salida, (5, 5), 0)
    ruido = rng.normal(0, 8, salida.shape)
    return np.clip(salida.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def _contorno_de(imagen_gris):
    _, binaria = cv2.threshold(
        imagen_gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    return mayor if cv2.contourArea(mayor) > 200 else None


def _aumentar_referencia(imagen, rng):
    """Rota y escala una foto de referencia del alumno."""
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, (LADO, LADO))
    matriz = cv2.getRotationMatrix2D(
        (LADO / 2, LADO / 2), rng.uniform(0, 360), rng.uniform(0.6, 1.1)
    )
    rotada = cv2.warpAffine(imagen, matriz, (LADO, LADO), borderValue=255)
    return _deformar(rotada, rng)


def construir(clases, muestras_por_clase=300, semilla=0, directorio_referencias=None):
    """Devuelve (X, y, nombres) con los embeddings y su clase.

    X: matriz (n, embeddings.DIMENSION) de descriptores de forma.
    y: vector (n,) de indices de clase.
    nombres: lista de nombres de clase en el orden de los indices.
    """
    rng = np.random.default_rng(semilla)
    nombres = list(clases)

    # Fotos de referencia del alumno, si las hay, para aumentarlas tambien.
    fotos = {}
    directorio = directorio_referencias or config.DIR_REFERENCIAS
    for ruta in list(directorio.glob("*.png")) + list(directorio.glob("*.jpg")):
        if ruta.stem not in nombres:
            continue
        foto = cv2.imread(str(ruta))
        if foto is None:
            print(f"AVISO: no se pudo leer {ruta}, se ignora.")
            continue
        fotos.setdefault(ruta.stem, []).append(foto)

    vectores, etiquetas = [], []
    for indice, clase in enumerate(nombres):
        generadas = 0
        intentos = 0
        while generadas < muestras_por_clase and intentos < muestras_por_clase * 5:
            intentos += 1
            propias = fotos.get(clase)
            dibujada = None if propias and rng.random() < 0.5 else _dibujar(clase, rng)
            if dibujada is not None:
                imagen = _deformar(dibujada, rng)
            elif propias:
                # Mezcla: la mitad de las muestras salen de las imagenes de
                # referencia (y el total, si la clase no tiene generador).
                imagen = _aumentar_referencia(
                    propias[rng.integers(len(propias))], rng
                )
            else:
                raise ValueError(
                    f"la clase '{clase}' no tiene generador sintetico ni imagen "
                    f"en {directorio}: agregar una referencia con "
                    f"capturar_referencias.py"
                )

            contorno = _contorno_de(imagen)
            if contorno is None:
                continue
            vectores.append(embeddings.vector(contorno))
            etiquetas.append(indice)
            generadas += 1

        if generadas < muestras_por_clase:
            print(f"AVISO: solo se generaron {generadas}/{muestras_por_clase} "
                  f"muestras de '{clase}'.")

    if not vectores:
        raise ValueError(
            "no se pudo generar ninguna muestra: revisar vision/referencias/"
        )
    X = np.vstack(vectores).astype(np.float32)
    y = np.array(etiquetas, dtype=np.int32)
    return X, y, nombres
