"""Busca que combinacion de backend e indice de camara funciona en esta maquina.

Los backends candidatos salen del registro de OpenCV (los realmente compilados
en este build) en vez de una lista fija, asi el diagnostico sirve igual en
Linux, en Windows y en macOS.
"""

import sys

import cv2

from vision import camara as modulo_camara

INDICES = range(4)


def _backends():
    """Backends de camara compilados, mas CAP_ANY al final."""
    registrados = list(cv2.videoio_registry.getCameraBackends())
    if cv2.CAP_ANY not in registrados:
        registrados.append(cv2.CAP_ANY)
    return registrados


def main():
    print(f"OpenCV {cv2.__version__} en {sys.platform}")
    print("Backends de camara compilados:",
          ", ".join(modulo_camara.nombre_backend(b)
                    for b in cv2.videoio_registry.getCameraBackends()))
    print()

    encontradas = []
    for backend in _backends():
        nombre = modulo_camara.nombre_backend(backend)
        for indice in INDICES:
            try:
                cap = cv2.VideoCapture(indice, backend)
                abierta = cap.isOpened()
                leyo, cuadro = cap.read() if abierta else (False, None)
                cap.release()
            except cv2.error as e:
                print(f"{nombre:12} indice {indice}: excepcion {e}")
                continue

            if leyo and cuadro is not None:
                print(f"{nombre:12} indice {indice}: OK  "
                      f"{cuadro.shape[1]}x{cuadro.shape[0]}")
                encontradas.append((nombre, indice))
            elif abierta:
                print(f"{nombre:12} indice {indice}: abre pero no lee")
            else:
                print(f"{nombre:12} indice {indice}: no abre")

    print()
    if not encontradas:
        print("Ninguna combinacion funciono.")
        print("Sin camara igual se puede usar todo el pipeline con:")
        print("    python main.py --imagen pruebas/escena.png")
        return 1

    print("Funciona:", encontradas)
    nombre, indice = encontradas[0]
    print(f"\nPara usarla:\n    python main.py --camara {indice} "
          f"--backend {nombre.lower().replace('cap_', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
