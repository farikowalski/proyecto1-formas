"""Apertura de la webcam de forma portable entre sistemas operativos.

OpenCV no elige solo el backend de captura: si se le pide uno que no esta
compilado, `VideoCapture` NO lanza ninguna excepcion ni imprime nada, y
devuelve un objeto cerrado. Ese era el error del proyecto: pedia
`cv2.CAP_DSHOW` (DirectShow, exclusivo de Windows), asi que en Linux la
camara "no abria" aunque /dev/video0 funcionara perfecto.

La solucion es probar los backends que corresponden a la plataforma, en
orden, y quedarse con el primero que abra y ademas devuelva un cuadro.
"""

import sys

import cv2

# Nombres aceptados por --backend. "auto" deja que este modulo decida.
BACKENDS_POR_NOMBRE = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


def nombre_backend(backend):
    """Nombre legible del backend ('V4L2', 'DSHOW', ...)."""
    try:
        return cv2.videoio_registry.getBackendName(backend)
    except cv2.error:
        return str(backend)


def backends_preferidos():
    """Backends a probar, en orden, segun el sistema operativo."""
    if sys.platform.startswith("win"):
        preferidos = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    elif sys.platform == "darwin":
        preferidos = [cv2.CAP_AVFOUNDATION]
    else:
        preferidos = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]

    # Se descartan los que este build de OpenCV no trae compilados.
    usables = [b for b in preferidos if cv2.videoio_registry.hasBackend(b)]
    # CAP_ANY es la autodeteccion de OpenCV: siempre va al final y nunca se
    # filtra, porque hasBackend(CAP_ANY) devuelve False aunque siempre sirva.
    return usables + [cv2.CAP_ANY]


def abrir(indice, ancho=None, alto=None, backend=None):
    """Abre la camara y devuelve (captura, nombre_backend) o (None, None).

    Se prueba cada backend candidato hasta que uno abra y lea un cuadro: que
    `isOpened()` diga True no alcanza, hay dispositivos que abren y despues
    fallan al leer (es el caso del segundo nodo /dev/videoN de muchas webcams).
    """
    candidatos = [backend] if backend is not None else backends_preferidos()

    for candidato in candidatos:
        captura = cv2.VideoCapture(indice, candidato)
        if not captura.isOpened():
            captura.release()
            continue

        # Los set() recien tienen sentido sobre una captura ya abierta.
        if ancho:
            captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        if alto:
            captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)

        ok, _ = captura.read()
        if ok:
            return captura, nombre_backend(candidato)
        captura.release()

    return None, None


def resolucion(captura):
    """Resolucion que la camara acepto realmente, que puede no ser la pedida."""
    return (
        int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
