from gl import *


# Valores aleatorios
import random


# Desplegar resultado
# Referencia: https://www.geeksforgeeks.org/python-pil-image-show-method/
from PIL import Image

# Colores
red = color(1, 0, 0)
wh = color(1, 1, 1)

# Definicion de variables para versión aleatoria
firstValue = random.random()
secondValue = random.random()
thirdValue = random.random()

firstColor = random.randint(0, 1)
secondColor = random.randint(0, 1)
thirdColor = random.randint(0, 1)

firstX = random.randint(0, 1)
firstY = random.randint(0, 1)


def SR6(filename, shot):
    r = Render(720, 540)
    r.shaderUsed = Shader.flatShading
    # Look at:
    OjitosUwU = V3(0, -30, -30)
    # Tamanio del SONIC
    Soniquin = V3(0.62, 0.62, 0.62)
    # Shot definition

    # ------------------ 1 ------------------
    if shot == 1:
        r.glLookAt(V3(0, -25, -25), V3(0, -10, 0))

        r.glModel(
            "sonic.obj",
            translation=V3(0, -25, -25),
            scalationFactor=Soniquin,
        )

        filename = filename + "MS.bmp"
        # https://www.studiobinder.com/blog/medium-shot-examples/

        r.glFinish(filename)
        im = Image.open(filename)
        im.show()

    if shot == 2:
        r.glLookAt(V3(0, -5, -5), V3(0, 0, 7))

        r.glModel(
            "sonic.obj",
            translation=V3(0, -5, -5),
            scalationFactor=Soniquin,
        )

        filename = filename + "LS.bmp"
        # https://www.studiobinder.com/blog/low-angle-shot-camera-movement-angle/

        r.glFinish(filename)
        im = Image.open(filename)
        im.show()

    if shot == 3:
        r.glLookAt(OjitosUwU, V3(0, -5, 0))

        r.glModel(
            "sonic.obj",
            translation=OjitosUwU,
            scalationFactor=Soniquin,
        )

        filename = filename + "HA.bmp"
        # https://www.studiobinder.com/blog/high-angle-shot-camera-movement-angle/

        r.glFinish(filename)
        im = Image.open(filename)
        im.show()
    if shot == 4:
        r.glLookAt(V3(5, 5, -5), V3(0, 0, 15))

        r.glModel(
            "sonic.obj",
            translation=V3(5, 5, -5),
            scalationFactor=Soniquin,
        )

        filename = filename + "DA.bmp"
        # https://www.studiobinder.com/blog/low-angle-shot-camera-movement-angle/

        r.glFinish(filename)
        im = Image.open(filename)
        im.show()
