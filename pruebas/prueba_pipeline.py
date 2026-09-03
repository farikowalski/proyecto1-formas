"""Prueba de extremo a extremo sin camara.

Genera una escena sintetica con las tres formas conocidas mas dos intrusas
(una estrella y un hexagono, que deben quedar como "desconocido") y verifica
que el pipeline completo las detecte y clasifique con los dos metodos.

    python -m pruebas.prueba_pipeline

Las imagenes se escriben en pruebas/salida/, que esta en .gitignore: la prueba
no pisa ningun archivo versionado.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import (  # noqa: E402
    anotacion,
    clasificador as modulo_clasificador,
    config,
    modelo as modulo_modelo,
    procesamiento,
    referencias,
)

DIR_SALIDA = Path(__file__).resolve().parent / "salida"
ESCENA = DIR_SALIDA / "escena.png"
ANOTADA = DIR_SALIDA / "escena_anotada.png"

# Parametros propios de la prueba: no se leen de los valores iniciales de las
# barras, para que ajustar la interfaz no rompa la suite.
AREA_MINIMA = 1500
TOLERANCIA = 1.0                  # umbrales por clase tal cual
DISTANCIA_MAXIMA_KNN = 0.50


def _poligono(centro, radio, lados, rotacion=-np.pi / 2):
    angulos = np.linspace(0, 2 * np.pi, lados, endpoint=False) + rotacion
    puntos = np.stack(
        [centro[0] + radio * np.cos(angulos), centro[1] + radio * np.sin(angulos)],
        axis=1,
    )
    return puntos.astype(np.int32)


def _estrella(centro, radio, puntas=5):
    angulos = np.linspace(0, 2 * np.pi, puntas * 2, endpoint=False) - np.pi / 2
    radios = np.where(np.arange(puntas * 2) % 2 == 0, radio, radio * 0.42)
    puntos = np.stack(
        [centro[0] + radios * np.cos(angulos), centro[1] + radios * np.sin(angulos)],
        axis=1,
    )
    return puntos.astype(np.int32)


def generar_escena():
    """Hoja blanca con cinco figuras negras, como el ambiente controlado."""
    escena = np.full((520, 1120, 3), 245, dtype=np.uint8)
    negro = (25, 25, 25)

    cv2.fillPoly(escena, [_poligono((130, 190), 105, 3)], negro)
    cv2.fillPoly(escena, [_poligono((350, 190), 110, 4, np.pi / 4)], negro)
    cv2.circle(escena, (570, 190), 90, negro, -1)
    cv2.fillPoly(escena, [_estrella((790, 190), 95)], negro)
    cv2.fillPoly(escena, [_poligono((1000, 190), 95, 6, 0.0)], negro)

    # Ruido: manchitas que el filtro de area y la morfologia deben descartar.
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y = rng.integers(20, 1100), rng.integers(350, 500)
        cv2.circle(escena, (int(x), int(y)), int(rng.integers(1, 4)), (30, 30, 30), -1)

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ESCENA), escena)
    return escena


def main():
    escena = generar_escena()
    print(f"Escena generada: {ESCENA}")

    parametros = {
        "umbral": config.UMBRAL_INICIAL,
        "auto_otsu": True,
        "kernel": 5,
        "area_minima": AREA_MINIMA,
        "fondo_claro": True,
    }
    pasos = procesamiento.procesar(escena, parametros)
    contornos = pasos["contornos"]
    print(f"Contornos utiles detectados: {len(contornos)} (esperado 5)")
    if len(contornos) != 5:
        print("FALLA: el filtrado de ruido o de area no funciono")
        return 1

    # Los contornos vienen en orden arbitrario: se ordenan por posicion en x.
    contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])
    esperado = ["triangulo", "cuadrado", "circulo", None, None]
    nombres_figura = ["triangulo", "cuadrado", "circulo", "estrella", "hexagono"]

    refs = referencias.cargar()
    backends = {
        "matchShapes": (
            modulo_clasificador.ClasificadorMatchShapes(refs), TOLERANCIA
        )
    }
    if modulo_modelo.existe():
        backends["embedding k-NN"] = (
            modulo_clasificador.ClasificadorEmbedding(
                modulo_modelo.ClasificadorKNN.cargar()
            ),
            DISTANCIA_MAXIMA_KNN,
        )
    else:
        print("FALLA: no hay modelo entrenado (modelo/modelo.npz). "
              "Ejecutar 'python entrenar.py'.")
        return 1

    fallas = 0
    resultados_por_backend = {}
    for nombre_backend, (clasificador, umbral) in backends.items():
        print(f"\n--- {nombre_backend} (umbral {umbral}) ---")
        resultados = modulo_clasificador.clasificar_todos(contornos, clasificador, umbral)
        resultados_por_backend[nombre_backend] = resultados
        for indice, ((_, nombre, distancia), esperada) in enumerate(
            zip(resultados, esperado)
        ):
            if nombre == esperada:
                marca = "OK "
            else:
                marca = "MAL"
                fallas += 1
            print(
                f"  {marca} {nombres_figura[indice]:<10} predicho={str(nombre):<12} "
                f"esperado={str(esperada):<12} distancia={distancia:.4f}"
            )

    # La imagen anotada se genera con matchShapes, que es el metodo del enunciado.
    salida = escena.copy()
    resultados = resultados_por_backend["matchShapes"]
    parametros.update({
        "umbral_usado": int(pasos["umbral_usado"]),
        "metodo": config.METODO_MATCHSHAPES,
        "tolerancia": TOLERANCIA,
        "roi_relativa": (0.0, 0.0, 1.0, 1.0),
    })
    anotacion.anotar(salida, resultados, tope=config.ALTO_PANEL)
    anotacion.panel_estado(salida, parametros, resultados, clases=sorted(refs))
    cv2.imwrite(str(ANOTADA), salida)
    print(f"\nImagen anotada: {ANOTADA}")

    if fallas:
        print(f"\nFALLARON {fallas} clasificaciones")
        return 1
    print("\nTodas las clasificaciones fueron correctas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
