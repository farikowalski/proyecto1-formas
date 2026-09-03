"""Captura las imagenes de referencia de tus propios objetos con la webcam.

Poner el objeto solo sobre el fondo claro, ajustar el umbral hasta ver la
figura limpia en la ventana binaria y apretar la tecla de la clase:

    1  guarda vision/referencias/triangulo.png
    2  guarda vision/referencias/cuadrado.png
    3  guarda vision/referencias/circulo.png
    q  salir

Se guarda el recorte binario del objeto mas grande de la escena, invertido
para que quede negro sobre blanco como el resto de las referencias.
Despues de capturar conviene volver a entrenar: python entrenar.py
"""

import sys

import cv2
import numpy as np

from vision import config, interfaz, procesamiento

TECLAS = {ord("1"): "triangulo", ord("2"): "cuadrado", ord("3"): "circulo"}
MARGEN = 20


def _recorte_binario(binaria, contorno):
    """Recorta el objeto y lo devuelve en negro sobre fondo blanco."""
    x, y, w, h = cv2.boundingRect(contorno)
    lienzo = np.full((h + 2 * MARGEN, w + 2 * MARGEN), 255, dtype=np.uint8)
    cv2.drawContours(lienzo, [contorno - (x - MARGEN, y - MARGEN)], -1, 0, -1)
    return lienzo


def main():
    captura = cv2.VideoCapture(config.INDICE_CAMARA, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not captura.isOpened():
        print("No se pudo abrir la camara.")
        return 1

    interfaz.crear_controles()
    config.DIR_REFERENCIAS.mkdir(parents=True, exist_ok=True)
    print(__doc__)

    try:
        while True:
            ok, cuadro = captura.read()
            if not ok:
                break

            parametros = interfaz.leer_controles()
            pasos = procesamiento.procesar(cuadro, parametros)
            contornos = pasos["contornos"]
            mayor = max(contornos, key=cv2.contourArea) if contornos else None

            vista = cuadro.copy()
            if mayor is not None:
                cv2.drawContours(vista, [mayor], -1, (0, 220, 0), 2)
                cv2.putText(
                    vista, "1=triangulo  2=cuadrado  3=circulo  q=salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2,
                )
            cv2.imshow("Captura de referencias", vista)
            cv2.imshow("Binaria", pasos["limpia"])

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla in TECLAS:
                if mayor is None:
                    print("No hay ningun objeto detectado todavia.")
                    continue
                nombre = TECLAS[tecla]
                ruta = config.DIR_REFERENCIAS / f"{nombre}.png"
                cv2.imwrite(str(ruta), _recorte_binario(pasos["limpia"], mayor))
                print(f"Guardada referencia de '{nombre}' en {ruta}")
    finally:
        captura.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
