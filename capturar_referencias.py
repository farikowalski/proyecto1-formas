"""Captura las imagenes de referencia de tus propios objetos con la webcam.

El enunciado pide elegir al menos tres objetos reales y usar UNA IMAGEN DE
CADA UNO como referencia. Esta herramienta es la que las saca: poner el objeto
solo sobre el fondo claro, ajustar el umbral hasta ver la figura limpia en la
ventana binaria y apretar la tecla de la clase.

    1..N  guarda vision/referencias/<clase>.png (segun config.CLASES)
    q     salir

Se guarda el recorte binario del objeto mas grande de la escena, invertido
para que quede negro sobre blanco como el resto de las referencias.
Despues de capturar conviene volver a entrenar: python entrenar.py
"""

import argparse
import sys

import cv2
import numpy as np

from vision import camara as modulo_camara, config, interfaz, procesamiento

MARGEN = 20

# Las teclas salen de config.CLASES: agregar una clase no obliga a tocar esto.
TECLAS = {ord(str(i + 1)): clase for i, clase in enumerate(config.CLASES[:9])}
AYUDA = "  ".join(f"{i + 1}={clase}" for i, clase in enumerate(config.CLASES[:9]))


def _recorte_binario(contorno):
    """Silueta del objeto, negra sobre fondo blanco, con un margen."""
    x, y, w, h = cv2.boundingRect(contorno)
    lienzo = np.full((h + 2 * MARGEN, w + 2 * MARGEN), 255, dtype=np.uint8)
    cv2.drawContours(lienzo, [contorno - (x - MARGEN, y - MARGEN)], -1, 0, -1)
    return lienzo


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camara", type=int, default=config.INDICE_CAMARA,
                        help="indice de camara (en Linux, N = /dev/videoN)")
    parser.add_argument("--backend", default="auto",
                        choices=sorted(modulo_camara.BACKENDS_POR_NOMBRE),
                        help="backend de captura; 'auto' elige segun el sistema")
    args = parser.parse_args()

    captura, backend = modulo_camara.abrir(
        args.camara,
        config.ANCHO_CAMARA,
        config.ALTO_CAMARA,
        modulo_camara.BACKENDS_POR_NOMBRE[args.backend],
    )
    if captura is None:
        print(f"No se pudo abrir la camara {args.camara}.")
        print("Probar 'python diagnostico_camara.py'.")
        return 1
    print(f"Camara {args.camara} abierta con el backend {backend}.")

    # Aca solo hacen falta las barras de segmentacion.
    interfaz.crear_controles(incluir_clasificacion=False)
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
            # La ayuda va SIEMPRE, sobre todo cuando no se detecta nada: es
            # justo el momento en que el usuario no sabe que hacer.
            cv2.putText(vista, f"{AYUDA}  q=salir", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
            if mayor is not None:
                cv2.drawContours(vista, [mayor], -1, (0, 220, 0), 2)
                x, y, w, h = cv2.boundingRect(mayor)
                cv2.putText(vista, f"area={int(cv2.contourArea(mayor))}px",
                            (x, max(y - 8, 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(vista, "no se detecta ningun objeto: ajusta el "
                            "umbral o el area minima", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

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
                if cv2.imwrite(str(ruta), _recorte_binario(mayor)):
                    print(f"Guardada referencia de '{nombre}' en {ruta}")
                else:
                    print(f"No se pudo escribir {ruta}")
    finally:
        captura.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
