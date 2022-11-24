"""
Created by:
Alejandro Gomez

Features:
Loading an object from a file
Loads textures from file
Generate shader lightining
"""


import struct

"""
Class made to open an object from a file
"""


class ObjectOpener:
    def __init__(self, filename):
        with open(filename, "r") as file:
            self.lines = file.read().splitlines()
        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.faces = []
        self.glLines1()

    def glLines1(self):
        for line in self.lines:
            try:
                prefix, value = line.split(" ", 1)
            except:
                continue
            if prefix == "v":
                self.vertices.append(list(map(float, value.split(" "))))

            elif prefix == "vt":
                self.texcoords.append(list(map(float, value.split(" "))))
            elif prefix == "vn":
                self.normals.append(list(map(float, value.split(" "))))
            elif prefix == "f":
                self.faces.append(
                    [list(map(int, vert.split("/"))) for vert in value.split(" ")]
                )


"""
Class made to use a texture from a file
"""


class Texture(object):
    def __init__(self, filename):
        with open(filename, "rb") as file:
            file.seek(10)
            hS = struct.unpack("=l", file.read(4))[0]
            file.seek(18)
            self.width = struct.unpack("=l", file.read(4))[0]
            self.height = struct.unpack("=l", file.read(4))[0]
            file.seek(hS)
            self.pixels = []
            for y in range(self.height):
                pR = []
                for x in range(self.width):
                    b = ord(file.read(1)) / 255
                    g = ord(file.read(1)) / 255
                    r = ord(file.read(1)) / 255
                    pR.append([r, g, b])
                self.pixels.append(pR)

    def getColor(self, u, v):
        if 0 <= u <= 1 and 0 <= v <= 1:
            return self.pixels[int(v * self.height)][int(u * self.width)]
        else:
            return None


"""
Referencia:
https://j2logo.com/args-y-kwargs-en-python/
https://es.wikipedia.org/wiki/Sombreado_plano
https://graphics.fandom.com/wiki/Flat_shading
https://www.giantbomb.com/flat-shading/3015-2277/
https://cglearn.codelight.eu/pub/computer-graphics/shading-and-lighting
"""


class Shader:
    @staticmethod
    def flatShading(render, **kwargs) -> None:
        u, v, w = kwargs["baryCoords"]
        b, g, r = kwargs["colorU"]
        tA, tB, tC = kwargs["textureCoords"]
        tN = kwargs["triangleNormal"]

        b = b / 255
        g = g / 255
        r = r / 255

        if render.textureUsed:

            tempValue = tA[0] * u
            tempValue2 = tB[0] * v
            tempValue3 = tC[0] * w
            tempValue4 = tA[1] * u
            tempValue5 = tB[1] * v
            tempValue6 = tC[1] * w

            tU = tempValue + tempValue2 + tempValue3
            tV = tempValue4 + tempValue5 + tempValue6

            colorizarTextura = render.textureUsed.getColor(tU, tV)
            b *= colorizarTextura[2]
            g *= colorizarTextura[1]
            r *= colorizarTextura[0]

        luzDirecta = [render.luzDirecta.x, render.luzDirecta.y, render.luzDirecta.z]
        invertedLight = [(-i) for i in luzDirecta]

        result = 0
        for i in range(0, len(tN)):
            result += tN[i] * invertedLight[i]
        finalValue = result

        b *= finalValue
        g *= finalValue
        r *= finalValue

        if finalValue > 0:
            return r, g, b
        return 0, 0, 0

    @staticmethod
    def neptuneShader(render, **kwargs) -> None:

        u, v, w = kwargs["baryCoords"]

        b, g, r = kwargs["colorU"]
        tA, tB, tC = kwargs["textureCoords"]
        tN = kwargs["triangleNormal"]

        r, g, b = (46 / 255, 57 / 255, 146 / 255)

        luzDirecta = [render.luzDirecta.x, render.luzDirecta.y, render.luzDirecta.z]
        invertedLight = [(-i + 0.3) for i in luzDirecta]

        result = 0
        for i in range(0, len(tN)):
            result += tN[i] * invertedLight[i]
        finalValue = result + 0.1

        b *= finalValue
        g *= finalValue
        r *= finalValue

        if finalValue < 0.30 and finalValue > 0.10 and finalValue < 0.35:
            r, g, b = (0, 0, 0.3)

        if finalValue < 0.10 and finalValue > 0.05:
            r, g, b = (0, 0, 0.5)

        if finalValue < 0.05:
            r, g, b = (0, 0, 0.8)

        if finalValue > 0.35 and finalValue < 0.37:
            r, g, b = (1, 1, 1)

        if finalValue > 0.37 and finalValue < 0.39:
            r, g, b = (0.2, 0.2, 0.9)

        if finalValue > 0.39 and finalValue < 0.41:
            r, g, b = (0.4, 0, 0.9)

        if finalValue > 0.41 and finalValue < 0.43:
            r, g, b = (0.2, 0, 0.9)

        if finalValue > 0.43 and finalValue < 0.45:
            r, g, b = (0.2, 0.5, 0.9)

        if finalValue > 0.45 and finalValue < 0.47:
            r, g, b = (0.77, 1, 1)

        if finalValue > 0.47 and finalValue < 0.49:
            r, g, b = (0.32, 0.55, 1)

        if finalValue > 0.49 and finalValue < 0.51:
            r, g, b = (0.32, 0.52, 1)

        if finalValue > 0.51 and finalValue < 0.53:
            r, g, b = (0.32, 0.53, 1)

        if finalValue > 0.53 and finalValue < 0.55:
            r, g, b = (0.32, 0.58, 1)

        if finalValue > 0.55 and finalValue < 0.57:
            r, g, b = (0.32, 0.58, 1)

        if finalValue > 0.57 and finalValue < 0.59:
            r, g, b = (0.32, 0.48, 1)

        if finalValue > 0.59 and finalValue < 0.61:
            r, g, b = (0.32, 0.58, 1)

        if finalValue > 0.61 and finalValue < 0.63:
            r, g, b = (0.32, 0.68, 1)

        if finalValue > 0.63 and finalValue < 0.65:
            r, g, b = (0.32, 0.68, 1)

        if finalValue > 0.65 and finalValue < 0.90:
            r, g, b = (0.2, 0.18, 1)

        if finalValue > 0.80 and finalValue < 0.85:
            r, g, b = (0.01, 0.01, 1)
        if finalValue > 0.91 and finalValue < 0.98:
            r, g, b = (0.8, 0.8, 1)

        if finalValue > 0.65 and finalValue < 0.67:
            r, g, b = (0.32, 0.18, 1)

        if finalValue > 0.67 and finalValue < 0.69:
            r, g, b = (0.30, 0.21, 1)

        # if finalValue > 0.69 and finalValue < 0.71:

        # if finalValue > 0.71 and finalValue < 0.73:
        if finalValue > 0:
            return r, g, b
        return 0, 0, 0
