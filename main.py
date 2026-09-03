"""Deteccion y clasificacion de formas en tiempo real.

Proyecto 1 - Vision artificial - Universidad Austral.

Uso:
    python main.py                              # camara web
    python main.py --imagen pruebas/escena.png  # imagen fija, sin camara
    python main.py --metodo embedding           # extra: k-NN sobre embeddings
    python main.py --camara 1 --backend v4l2    # elegir camara y backend
    python main.py --generar-referencias

Teclas:
    q / ESC  salir
    espacio  pausar y reanudar
    m        alternar matchShapes / k-NN
    g        guardar la imagen anotada en capturas/
"""

import argparse
import os
import sys
import time

import cv2

from vision import (
    anotacion,
    camara as modulo_camara,
    clasificador as modulo_clasificador,
    config,
    interfaz,
    modelo as modulo_modelo,
    procesamiento,
    referencias,
)


def _hay_entorno_grafico():
    """En Linux sin DISPLAY, OpenCV aborta el proceso al crear una ventana."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _cargar_clasificadores():
    """Devuelve {metodo: clasificador} con los backends disponibles."""
    disponibles = {}

    refs = referencias.cargar()
    if not refs:
        print("No hay referencias en vision/referencias/: se generan las ideales.")
        referencias.generar_sinteticas()
        refs = referencias.cargar()
    print("Referencias cargadas:", ", ".join(sorted(refs)))

    clasificador_ms = modulo_clasificador.ClasificadorMatchShapes(refs)
    umbrales = ", ".join(
        f"{nombre} {clasificador_ms.umbral_de(nombre):.3f}" for nombre in sorted(refs)
    )
    print("Umbral de validez por clase:", umbrales)
    disponibles[config.METODO_MATCHSHAPES] = clasificador_ms

    if modulo_modelo.existe():
        modelo = modulo_modelo.ClasificadorKNN.cargar()
        print(f"Modelo k-NN cargado (k={modelo.k}, "
              f"{modelo.X.shape[0]} embeddings): {', '.join(modelo.nombres)}")
        disponibles[config.METODO_EMBEDDING] = (
            modulo_clasificador.ClasificadorEmbedding(modelo)
        )
    else:
        print("No hay modelo entrenado (modelo/modelo.npz). "
              "Ejecutar 'python entrenar.py' para usar el metodo k-NN.")
    return disponibles


def _analizar(cuadro, clasificador, parametros):
    """Procesa un cuadro y devuelve (imagen_anotada, pasos, resultados)."""
    roi_relativa = parametros["roi_relativa"]
    roi, desplazamiento = procesamiento.recortar_roi(cuadro, roi_relativa)
    pasos = procesamiento.procesar(roi, parametros)
    resultados = modulo_clasificador.clasificar_todos(
        pasos["contornos"], clasificador, parametros["umbral_efectivo"]
    )

    salida = cuadro.copy()
    if roi_relativa != (0.0, 0.0, 1.0, 1.0):
        # Se marca el rectangulo de la ROI: fuera de ahi no se busca nada.
        x, y = desplazamiento
        alto, ancho = roi.shape[:2]
        cv2.rectangle(salida, (x, y), (x + ancho, y + alto), (200, 200, 200), 1)
        cv2.putText(salida, "ROI", (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    anotacion.anotar(salida, resultados, desplazamiento, tope=config.ALTO_PANEL)
    return salida, pasos, resultados


def _guardar(imagen):
    config.DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
    ruta = config.DIR_CAPTURAS / f"captura_{int(time.time())}.png"
    if cv2.imwrite(str(ruta), imagen):
        print("Guardado:", ruta)
    else:
        print("No se pudo guardar en", ruta)


def _crear_ventanas():
    """Ventana principal mas las dos de pasos intermedios, sin apilarse."""
    for nombre, tamano, posicion in (
        (config.VENTANA_SALIDA, (960, 540), (480, 0)),
        (config.VENTANA_BINARIA, (460, 260), (480, 580)),
        (config.VENTANA_LIMPIA, (460, 260), (950, 580)),
    ):
        cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre, *tamano)
        cv2.moveWindow(nombre, *posicion)


def _bucle(leer_cuadro, clasificadores, metodo_inicial, es_video):
    interfaz.crear_controles(metodo_inicial)
    _crear_ventanas()
    cv2.moveWindow(interfaz.VENTANA_CONTROLES, 0, 0)

    clases = sorted({c for cl in clasificadores.values() for c in cl.clases})
    ultimo = time.time()
    fps = None
    pausado = False
    cuadro = None
    fallos = 0
    aviso_metodo = None

    while True:
        if not pausado or cuadro is None:
            ok, nuevo = leer_cuadro()
            if ok:
                cuadro = nuevo
                fallos = 0
            elif es_video:
                # Las webcams USB pierden cuadros sueltos: no es motivo para
                # terminar la demo, se reintenta un rato antes de rendirse.
                fallos += 1
                if fallos >= config.MAX_FALLOS_CAMARA:
                    print("Se perdio la senal de la camara.")
                    break
                if cuadro is None:
                    if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                        break
                    continue
            else:
                pausado = True
                if cuadro is None:
                    break

        parametros = interfaz.leer_controles()

        # El metodo elegido puede no estar disponible (por ejemplo, k-NN sin
        # modelo entrenado). Se resuelve el clasificador ANTES de decidir que
        # umbral aplicar, porque cada metodo usa su propia escala.
        clasificador = clasificadores.get(parametros["metodo"])
        if clasificador is None:
            clasificador = next(iter(clasificadores.values()))
            if aviso_metodo != parametros["metodo"]:
                aviso_metodo = parametros["metodo"]
                print(f"El metodo '{parametros['metodo']}' no esta disponible; "
                      f"se usa '{clasificador.nombre_metodo}'.")
            parametros["metodo"] = clasificador.nombre_metodo

        if parametros["metodo"] == config.METODO_MATCHSHAPES:
            parametros["umbral_efectivo"] = parametros["tolerancia"]
        else:
            parametros["umbral_efectivo"] = parametros["distancia_maxima_knn"]
            parametros["distancia_maxima"] = parametros["distancia_maxima_knn"]

        salida, pasos, resultados = _analizar(cuadro, clasificador, parametros)

        if es_video:
            ahora = time.time()
            instantaneo = 1.0 / max(ahora - ultimo, 1e-6)
            fps = instantaneo if fps is None else 0.9 * fps + 0.1 * instantaneo
            ultimo = ahora

        parametros["umbral_usado"] = int(pasos["umbral_usado"])
        interfaz.sincronizar_umbral(pasos["umbral_usado"])
        anotacion.panel_estado(salida, parametros, resultados, fps, clases)
        anotacion.ayuda_teclas(salida)
        if pausado:
            cv2.putText(salida, "PAUSA", (salida.shape[1] // 2 - 60, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow(config.VENTANA_SALIDA, salida)
        cv2.imshow(config.VENTANA_BINARIA, pasos["binaria"])
        cv2.imshow(config.VENTANA_LIMPIA, pasos["limpia"])

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("g"):
            _guardar(salida)
        if tecla == ord("m"):
            interfaz.alternar_metodo()
        if tecla == ord(" "):
            pausado = not pausado

        # Si el usuario cierra la ventana principal con la X, se termina
        # ordenadamente en vez de seguir dibujando en el vacio.
        if cv2.getWindowProperty(config.VENTANA_SALIDA, cv2.WND_PROP_VISIBLE) < 1:
            break
        if not interfaz.existe_ventana():
            interfaz.crear_controles(parametros["metodo"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--imagen", help="procesa una imagen fija en lugar de la camara")
    parser.add_argument("--camara", type=int, default=config.INDICE_CAMARA,
                        help="indice de camara (en Linux, N = /dev/videoN)")
    parser.add_argument("--backend", default="auto",
                        choices=sorted(modulo_camara.BACKENDS_POR_NOMBRE),
                        help="backend de captura; 'auto' elige segun el sistema")
    parser.add_argument(
        "--metodo",
        choices=[config.METODO_MATCHSHAPES, config.METODO_EMBEDDING],
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

    if not _hay_entorno_grafico():
        print("No hay entorno grafico (DISPLAY/WAYLAND_DISPLAY): "
              "este programa necesita ventanas para las barras de desplazamiento.")
        return 1

    clasificadores = _cargar_clasificadores()
    print("\nTeclas: q salir | espacio pausa | m cambiar metodo | g guardar\n")

    if args.imagen:
        imagen = cv2.imread(args.imagen)
        if imagen is None:
            print("No se pudo abrir la imagen:", args.imagen)
            return 1
        try:
            _bucle(lambda: (True, imagen), clasificadores, args.metodo, es_video=False)
        finally:
            cv2.destroyAllWindows()
        return 0

    captura, backend = modulo_camara.abrir(
        args.camara,
        config.ANCHO_CAMARA,
        config.ALTO_CAMARA,
        modulo_camara.BACKENDS_POR_NOMBRE[args.backend],
    )
    if captura is None:
        print(f"No se pudo abrir la camara {args.camara} "
              f"(backend pedido: {args.backend}).")
        print("Probar 'python diagnostico_camara.py' para ver que combinaciones "
              "funcionan, o 'python main.py --imagen pruebas/escena.png' para "
              "usar el sistema sin camara.")
        return 1

    ancho, alto = modulo_camara.resolucion(captura)
    print(f"Camara {args.camara} abierta con el backend {backend} a {ancho}x{alto}.")
    try:
        _bucle(captura.read, clasificadores, args.metodo, es_video=True)
    finally:
        captura.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
