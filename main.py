"""Deteccion y clasificacion de formas en tiempo real.

Proyecto 1 - Vision artificial - Universidad Austral.

Uso:
    python main.py                      # camara web
    python main.py --imagen escena.png  # imagen fija, sin camara
    python main.py --metodo matchshapes
    python main.py --generar-referencias

Teclas:
    q / ESC  salir
    g        guardar la imagen anotada en capturas/
    espacio  pausar y reanudar
"""

import argparse
import sys
import time

import cv2

from vision import (
    anotacion,
    clasificador as modulo_clasificador,
    config,
    interfaz,
    modelo as modulo_modelo,
    procesamiento,
    referencias,
)

VENTANA_SALIDA = "Deteccion y clasificacion de formas"
VENTANA_BINARIA = "Paso intermedio - binaria"
VENTANA_LIMPIA = "Paso intermedio - morfologia"
DIR_CAPTURAS = config.RAIZ / "capturas"


def _cargar_clasificadores():
    """Devuelve {metodo: clasificador} con los backends disponibles."""
    disponibles = {}

    refs = referencias.cargar()
    if not refs:
        print("No hay referencias en vision/referencias/: se generan las ideales.")
        referencias.generar_sinteticas()
        refs = referencias.cargar()
    print("Referencias cargadas:", ", ".join(sorted(refs)))
    disponibles[modulo_clasificador.METODO_MATCHSHAPES] = (
        modulo_clasificador.ClasificadorMatchShapes(refs)
    )

    if modulo_modelo.existe():
        modelo = modulo_modelo.ClasificadorKNN.cargar()
        print(f"Modelo k-NN cargado (k={modelo.k}, "
              f"{modelo.X.shape[0]} embeddings): {', '.join(modelo.nombres)}")
        disponibles[modulo_clasificador.METODO_EMBEDDING] = (
            modulo_clasificador.ClasificadorEmbedding(modelo)
        )
    else:
        print("No hay modelo entrenado (modelo/modelo.npz). "
              "Ejecutar 'python entrenar.py' para usar el metodo k-NN.")
    return disponibles


def _analizar(cuadro, clasificador, parametros):
    """Procesa un cuadro y devuelve (imagen_anotada, pasos, resultados)."""
    roi, desplazamiento = procesamiento.recortar_roi(cuadro, config.ROI_RELATIVA)
    pasos = procesamiento.procesar(roi, parametros)
    resultados = modulo_clasificador.clasificar_todos(
        pasos["contornos"], clasificador, parametros["distancia_maxima"]
    )
    salida = cuadro.copy()
    if config.ROI_RELATIVA != (0.0, 0.0, 1.0, 1.0):
        x, y = desplazamiento
        alto, ancho = roi.shape[:2]
        cv2.rectangle(salida, (x, y), (x + ancho, y + alto), (200, 200, 200), 1)
    anotacion.anotar(salida, resultados, desplazamiento)
    return salida, pasos, resultados


def _guardar(imagen):
    DIR_CAPTURAS.mkdir(exist_ok=True)
    ruta = DIR_CAPTURAS / f"captura_{int(time.time())}.png"
    cv2.imwrite(str(ruta), imagen)
    print("Guardado:", ruta)


def _bucle(leer_cuadro, clasificadores, metodo_inicial, es_video):
    interfaz.crear_controles(metodo_inicial)
    ultimo = time.time()
    fps = 0.0
    pausado = False
    cuadro = None

    while True:
        if not pausado or cuadro is None:
            ok, nuevo = leer_cuadro()
            if not ok:
                if es_video:
                    print("No se pudo leer el cuadro de la camara.")
                    break
                pausado = True
            else:
                cuadro = nuevo

        parametros = interfaz.leer_controles()
        # Si el metodo elegido en el trackbar no esta disponible, se usa el otro.
        clasificador = clasificadores.get(parametros["metodo"])
        if clasificador is None:
            clasificador = next(iter(clasificadores.values()))
            parametros["metodo"] = clasificador.nombre_metodo

        salida, pasos, resultados = _analizar(cuadro, clasificador, parametros)

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - ultimo, 1e-6)
        ultimo = ahora

        parametros["umbral_usado"] = int(pasos["umbral_usado"])
        anotacion.panel_estado(
            salida, parametros, len(resultados), fps if es_video else None
        )

        cv2.imshow(VENTANA_SALIDA, salida)
        cv2.imshow(VENTANA_BINARIA, pasos["binaria"])
        cv2.imshow(VENTANA_LIMPIA, pasos["limpia"])

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("g"):
            _guardar(salida)
        if tecla == ord(" "):
            pausado = not pausado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", help="procesa una imagen fija en lugar de la camara")
    parser.add_argument("--camara", type=int, default=config.INDICE_CAMARA)
    parser.add_argument(
        "--metodo",
        choices=[
            modulo_clasificador.METODO_EMBEDDING,
            modulo_clasificador.METODO_MATCHSHAPES,
        ],
        default=config.METODO_INICIAL,
    )
    parser.add_argument(
        "--generar-referencias",
        action="store_true",
        help="crea las imagenes de referencia ideales y termina",
    )
    args = parser.parse_args()

    if args.generar_referencias:
        for ruta in referencias.generar_sinteticas():
            print("Creada:", ruta)
        return 0

    clasificadores = _cargar_clasificadores()

    if args.imagen:
        imagen = cv2.imread(args.imagen)
        if imagen is None:
            print("No se pudo abrir la imagen:", args.imagen)
            return 1
        _bucle(lambda: (True, imagen), clasificadores, args.metodo, es_video=False)
        cv2.destroyAllWindows()
        return 0

    captura = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not captura.isOpened():
        print(f"No se pudo abrir la camara {args.camara}.")
        return 1
    try:
        _bucle(captura.read, clasificadores, args.metodo, es_video=True)
    finally:
        captura.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
