"""Entrena el clasificador de formas sobre los embeddings de Hu.

    python entrenar.py                 # 600 muestras por clase
    python entrenar.py --muestras 800 --k 7

Guarda el modelo en modelo/modelo.npz e imprime exactitud y matriz de
confusion sobre un conjunto de prueba separado.
"""

import argparse
import sys

import numpy as np

from vision import config, dataset, modelo as modulo_modelo


def dividir(X, y, proporcion_prueba=0.25, semilla=0):
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(y))
    corte = int(len(y) * (1 - proporcion_prueba))
    entrena, prueba = orden[:corte], orden[corte:]
    return X[entrena], y[entrena], X[prueba], y[prueba]


def imprimir_confusion(matriz, nombres):
    ancho = max(len(n) for n in nombres) + 2
    print("\nMatriz de confusion (filas = real, columnas = predicho)")
    print(" " * ancho + "".join(n.rjust(ancho) for n in nombres))
    for i, nombre in enumerate(nombres):
        fila = "".join(str(v).rjust(ancho) for v in matriz[i])
        print(nombre.ljust(ancho) + fila)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestras", type=int, default=600,
                        help="muestras generadas por clase")
    parser.add_argument("--k", type=int, default=5, help="vecinos del k-NN")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    if args.muestras < 1 or args.k < 1:
        parser.error("--muestras y --k tienen que ser >= 1")

    print(f"Generando {args.muestras} muestras por clase para {config.CLASES}...")
    X, y, nombres = dataset.construir(
        config.CLASES, muestras_por_clase=args.muestras, semilla=args.semilla
    )
    print(f"Dataset: {X.shape[0]} embeddings de dimension {X.shape[1]}")

    X_ent, y_ent, X_pru, y_pru = dividir(X, y, semilla=args.semilla)
    modelo = modulo_modelo.ClasificadorKNN(k=args.k).entrenar(X_ent, y_ent, nombres)

    exactitud = modelo.evaluar(X_pru, y_pru)
    print(f"\nExactitud en prueba: {exactitud * 100:.2f}% "
          f"({len(y_pru)} muestras)")
    imprimir_confusion(modelo.matriz_confusion(X_pru, y_pru), nombres)

    # Distancia tipica dentro de la clase: sirve para elegir el umbral de rechazo.
    distancias = [modelo.predecir_vector(v)[1] for v in X_pru]
    print(f"\nDistancia al vecino mas cercano en prueba: "
          f"mediana={np.median(distancias):.3f}  p95={np.percentile(distancias, 95):.3f}")

    ruta = modelo.guardar()
    print(f"\nModelo guardado en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
