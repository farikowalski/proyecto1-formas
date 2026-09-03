"""Dibujo de las anotaciones sobre la imagen de salida.

El enunciado pide una ventana con la imagen original anotada con la
localizacion de los objetos, verde/color por clase para los reconocidos, rojo
para los desconocidos, y una etiqueta con el nombre. Se agrega ademas lo que
el enunciado llama "coordenadas en pixeles y datos de tamano": el centroide y
el area de cada objeto.
"""

import cv2

from . import config

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX
_ESCALA = 0.55
_GROSOR = 1


def color_de(nombre):
    if nombre is None:
        return config.COLOR_DESCONOCIDO
    return config.COLOR_POR_CLASE.get(nombre, config.COLOR_DEFECTO)


def _caja_texto(texto):
    (ancho, alto), _ = cv2.getTextSize(texto, _FUENTE, _ESCALA, _GROSOR + 1)
    return ancho + 8, alto + 8


def _se_pisan(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _ubicar(candidatas, ancho_caja, alto_caja, ocupadas, limites):
    """Primera posicion candidata que entra en el cuadro y no pisa otra caja."""
    ancho_img, alto_img, tope = limites
    for x, y in candidatas:
        x = max(0, min(x, ancho_img - ancho_caja))
        y = max(tope, min(y, alto_img - alto_caja))
        caja = (x, y, ancho_caja, alto_caja)
        if not any(_se_pisan(caja, otra) for otra in ocupadas):
            return caja
    # Si todas se pisan, se usa la primera igual: mejor superpuesta que ausente.
    x, y = candidatas[0]
    x = max(0, min(x, ancho_img - ancho_caja))
    y = max(tope, min(y, alto_img - alto_caja))
    return (x, y, ancho_caja, alto_caja)


def _dibujar_etiqueta(imagen, texto, caja, color):
    x, y, ancho, alto = caja
    cv2.rectangle(imagen, (x, y), (x + ancho, y + alto), color, -1)
    cv2.putText(imagen, texto, (x + 4, y + alto - 6), _FUENTE, _ESCALA,
                (0, 0, 0), _GROSOR + 1, cv2.LINE_AA)


def centroide(contorno):
    """Coordenadas en pixeles del centro de masa del contorno."""
    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contorno)
        return x + w // 2, y + h // 2
    return int(momentos["m10"] / momentos["m00"]), int(momentos["m01"] / momentos["m00"])


def anotar(imagen, resultados, desplazamiento=(0, 0), tope=0):
    """Dibuja contorno, rectangulo, centroide y etiquetas de cada objeto.

    resultados: lista de (contorno, nombre_o_None, distancia).
    desplazamiento: origen de la ROI dentro de la imagen completa.
    tope: filas superiores reservadas para el panel, que las etiquetas evitan.
    """
    dx, dy = desplazamiento
    alto_img, ancho_img = imagen.shape[:2]
    ocupadas = []

    for contorno, nombre, distancia in resultados:
        color = color_de(nombre)
        desplazado = contorno + (dx, dy)
        cv2.drawContours(imagen, [desplazado], -1, color, 2)

        x, y, w, h = cv2.boundingRect(desplazado)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 1)

        cx, cy = centroide(desplazado)
        cv2.drawMarker(imagen, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

        texto = "desconocido" if nombre is None else nombre
        texto += f" d={distancia:.4f}"
        detalle = f"({cx},{cy}) area={int(cv2.contourArea(desplazado))}px"

        for linea, candidatas in (
            # Arriba del rectangulo; si no entra, adentro; si no, abajo.
            (texto, [(x, y - _caja_texto(texto)[1] - 2), (x, y + 2), (x, y + h + 2)]),
            (detalle, [(x, y + h + 2), (x, y + 2)]),
        ):
            ancho_caja, alto_caja = _caja_texto(linea)
            caja = _ubicar(candidatas, ancho_caja, alto_caja, ocupadas,
                           (ancho_img, alto_img, tope))
            _dibujar_etiqueta(imagen, linea, caja, color)
            ocupadas.append(caja)

    return imagen


def _leyenda(imagen, clases_visibles):
    """Cuadraditos de color por clase, para que el criterio quede explicito."""
    alto_img, ancho_img = imagen.shape[:2]
    entradas = [(nombre, color_de(nombre)) for nombre in clases_visibles]
    entradas.append(("desconocido", config.COLOR_DESCONOCIDO))

    ancho = 10 + sum(
        24 + cv2.getTextSize(nombre, _FUENTE, 0.45, 1)[0][0]
        for nombre, _ in entradas
    )
    base = alto_img - 8
    # Franja oscura de fondo: sin esto el texto blanco desaparece sobre la
    # hoja clara, que es justo el fondo del ambiente controlado.
    cv2.rectangle(imagen, (0, base - 22), (min(ancho, ancho_img), alto_img),
                  (0, 0, 0), -1)

    x = 10
    y = base - 4
    for nombre, color in entradas:
        cv2.rectangle(imagen, (x, y - 11), (x + 14, y + 1), color, -1)
        cv2.putText(imagen, nombre, (x + 19, y), _FUENTE, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += 24 + cv2.getTextSize(nombre, _FUENTE, 0.45, 1)[0][0]


def panel_estado(imagen, parametros, resultados, fps=None, clases=()):
    """Franja superior con los parametros activos y el conteo por estado."""
    reconocidos = [nombre for _, nombre, _ in resultados if nombre is not None]
    desconocidos = len(resultados) - len(reconocidos)
    detalle = ", ".join(
        f"{clase} {reconocidos.count(clase)}"
        for clase in sorted(set(reconocidos))
    )

    if parametros.get("metodo") == config.METODO_MATCHSHAPES:
        validez = "tolerancia={:.2f}x sobre umbral por clase".format(
            parametros.get("tolerancia", 1.0)
        )
    else:
        validez = "dist_max={:.2f}".format(parametros.get("distancia_maxima", 0.0))

    roi = parametros.get("roi_relativa", (0.0, 0.0, 1.0, 1.0))
    completa = roi == (0.0, 0.0, 1.0, 1.0)

    lineas = [
        "umbral={} {}  kernel={}  area_min={}  ROI={}".format(
            parametros.get("umbral_usado", 0),
            "(Otsu)" if parametros.get("auto_otsu") else "(manual)",
            parametros.get("kernel"),
            parametros.get("area_minima"),
            "completa" if completa else "{:.0f}%x{:.0f}%".format(
                (roi[2] - roi[0]) * 100, (roi[3] - roi[1]) * 100),
        ),
        "metodo={}  {}".format(parametros.get("metodo", "-"), validez),
        "reconocidos={}{}  desconocidos={}{}".format(
            len(reconocidos),
            f" ({detalle})" if detalle else "",
            desconocidos,
            f"  fps={fps:.1f}" if fps is not None else "",
        ),
    ]

    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], config.ALTO_PANEL), (0, 0, 0), -1)
    for i, linea in enumerate(lineas):
        cv2.putText(imagen, linea, (10, 18 + i * 21), _FUENTE, _ESCALA,
                    (255, 255, 255), _GROSOR, cv2.LINE_AA)

    if clases:
        _leyenda(imagen, clases)
    return imagen


def ayuda_teclas(imagen):
    """Recordatorio de teclas, abajo a la derecha."""
    texto = "q salir | espacio pausa | m metodo | g guardar"
    (ancho, alto), _ = cv2.getTextSize(texto, _FUENTE, 0.45, 1)
    x = imagen.shape[1] - ancho - 12
    y = imagen.shape[0] - 38
    cv2.rectangle(imagen, (x - 6, y - alto - 6), (imagen.shape[1], y + 6), (0, 0, 0), -1)
    cv2.putText(imagen, texto, (x, y), _FUENTE, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return imagen
