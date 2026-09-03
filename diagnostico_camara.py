"""Prueba combinaciones de backend e indice para encontrar una camara usable."""
import cv2

BACKENDS = [
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_ANY", cv2.CAP_ANY),
]

print(f"OpenCV {cv2.__version__}\n")
encontradas = []
for nombre, backend in BACKENDS:
    for indice in range(4):
        try:
            cap = cv2.VideoCapture(indice, backend)
            abierta = cap.isOpened()
            leyo, cuadro = cap.read() if abierta else (False, None)
            cap.release()
        except Exception as e:
            print(f"{nombre:10} indice {indice}: excepcion {e}")
            continue
        if leyo and cuadro is not None:
            print(f"{nombre:10} indice {indice}: OK  {cuadro.shape[1]}x{cuadro.shape[0]}")
            encontradas.append((nombre, indice))
        elif abierta:
            print(f"{nombre:10} indice {indice}: abre pero no lee")
        else:
            print(f"{nombre:10} indice {indice}: no abre")

print()
if encontradas:
    print("Funciona:", encontradas)
else:
    print("Ninguna combinacion funciono.")
