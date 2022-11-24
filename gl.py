# Modulo para estructura de bytes
import math
import struct


# Para objetos
from ObjectOpener import *


# definiciones de estructuras bytes
def char(c):
    return struct.pack("=c", c.encode("ascii"))


def word(w):
    return struct.pack("=h", w)


def dword(d):
    return struct.pack("=l", d)


def color(r, g, b):
    return bytes([int(b * 255), int(g * 255), int(r * 255)])


# Constates de bytes
firstW = word(1)
word24 = word(24)
headerInfo1 = 14 + 40
Dcero = dword(0)
Wcero = word(0)
w40 = dword(40)

# Valores constantes a usar de colores
color1 = "white"
color2 = "black"
color3 = "blue"

# Constante de valor
floatInfinite = float("inf")

# funcion para colores a usar
def BasicColors(const):
    if const == "white":
        return color(1, 1, 1)
    if const == "black":
        return color(0, 0, 0)
    if const == "blue":
        return color(0, 1, 1)


"""
Operaciones útiles de matemática

Referencias: 
https://stackoverflow.com/questions/28253102/python-3-multiply-a-vector-by-a-matrix-without-numpy
https://stackoverflow.com/questions/10508021/matrix-multiplication-in-pure-python
https://mathinsight.org/matrix_vector_multiplication
https://www.mathsisfun.com/algebra/matrix-multiplying.html

"""


def VerifyIntegers(value):
    try:
        value = int(value)
    except:
        print("Error")


def matrixMultiplication4x4(matrix, vector):

    if len(matrix) != 4 and len(matrix[0]) != 4:
        return None
    valueFinal = [0, 0, 0, 0]

    v4 = [
        [vector.x],
        [vector.y],
        [vector.z],
        [vector.w],
    ]

    for x in range(0, len(matrix)):
        tempValue = 0
        for y in range(0, len(matrix[x])):
            tempVariable = matrix[x][y] * v4[y][0]
            tempValue += float(tempVariable)
        valueFinal[x] = tempValue
    return valueFinal


def matrixMultiplications(matrixList):
    if len(matrixList) <= 1:
        return None

    matrixResult = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in matrixList:
        if len(i) != len(i[0]):
            return None
    taman = len(i)
    matrixResult = [[0 for y in range(0, taman)] for x in range(0, taman)]
    matrixLength = len(matrixList)
    for i in range(0, matrixLength):
        matrixValue = matrixList[i] if i == 0 else matrixResult
        secondMatrix = matrixList[i + 1]
        for x in range(0, len(matrixValue)):
            for y in range(0, len(matrixValue)):
                r = 0
                for z in range(0, len(matrixValue)):
                    r += matrixValue[x][z] * secondMatrix[z][y]
                matrixResult[x][y] = float(r)
        if (i + 2) == len(matrixList):
            break
    return matrixResult


from collections import *

"""
Referncia de documentación de namedtuple

Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

Usado como referencia para los puntos dentro de el renderizado
"""
V2 = namedtuple("XY", ["x", "y"])
V3 = namedtuple("XYZ", ["x", "y", "z"])
V4 = namedtuple("XYZW", ["x", "y", "z", "w"])

# Funcion para pasar a coordenadas barimetricas
def baryCoords(A, B, C, P):
    areaTemp1 = B.y - C.y
    areaTemp2 = P.x - C.x
    areaTemp3 = C.x - B.x
    areaTemp4 = P.y - C.y
    areaTemp5 = C.y - A.y
    areaTemp6 = A.x - C.x
    areaTemp7 = A.y - C.y

    # areas a hacer
    temporalAreaP_B_C = areaTemp1 * areaTemp2 + areaTemp3 * areaTemp4
    temporalAreaP_A_C = areaTemp5 * areaTemp2 + areaTemp6 * areaTemp4
    temporalAreaA_B_C = areaTemp1 * areaTemp6 + areaTemp3 * areaTemp7

    try:
        temporalDivisionArea = temporalAreaP_B_C / temporalAreaA_B_C
        temporalDivisionArea2 = temporalAreaP_A_C / temporalAreaA_B_C
        u = temporalAreaP_B_C / temporalAreaA_B_C
        v = temporalAreaP_A_C / temporalAreaA_B_C
        w = 1 - temporalDivisionArea - temporalDivisionArea2

    except:
        return -1, -1, -1
    return u, v, w


class Render(object):
    # (05 puntos) Deben crear una función glInit() que inicialice cualquier objeto interno que requiera su software renderer
    def __init__(self, width, height):
        self.glCreateWindow(width, height)
        self.clearColor = BasicColors(color2)
        self.currentColor = color(1, 1, 1)

        self.shaderUsed = None
        self.textureUsed = None
        self.luzDirecta = V3(0, 0, -1)
        self.glViewMatrix()
        self.glClear()

    # (05 puntos) Deben crear una función glCreateWindow(width, height) que inicialice su framebuffer con
    # un tamaño (la imagen resultante va a ser de este tamaño).
    def glCreateWindow(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.glViewPort(0, 0, self.width, self.height)

    # (10 puntos)  Deben crear una función glViewPort(x, y, width, height) que defina el área de la imagen sobre la que se va a poder dibujar
    def glViewPort(self, x, y, width, height):
        self.viewPortX = int(x)
        self.viewPortY = int(y)
        self.viewPortWidth = int(width)
        self.viewPortHeight = int(height)

        halfW = width / 2
        halfH = height / 2
        xHalfW = x + halfW
        yHalfH = y + halfH

        self.viewPortM = [
            [halfW, 0, 0, xHalfW],
            [0, height / 2, 0, yHalfH],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ]

        self.glProMatrix()

    def glViewMatrix(self, translate=V3(0, 0, 0), rotate=V3(0, 0, 0)):
        self.cameraMatrix = self.glCreateMatrix(translate, rotate)

        if len(self.cameraMatrix) != len(self.cameraMatrix[0]):
            self.viewMatrix = None

        mIdentidad = [
            [1 if x == y else 0 for y in range(0, len(self.cameraMatrix))]
            for x in range(0, len(self.cameraMatrix))
        ]
        for i in range(len(self.cameraMatrix)):
            self.cameraMatrix[i].extend(mIdentidad[i])
        for i in range(len(self.cameraMatrix)):
            verMat = self.cameraMatrix[i][i]
            if verMat == 0:
                self.viewMatrix = None
            for j in range(i, 2 * len(self.cameraMatrix)):
                self.cameraMatrix[i][j] = self.cameraMatrix[i][j] / verMat
            for z in range(len(self.cameraMatrix)):
                if z == i or self.cameraMatrix[z][i] == 0:
                    continue
                xsss = self.cameraMatrix[z][i]
                for j in range(i, 2 * len(self.cameraMatrix)):
                    self.cameraMatrix[z][j] -= xsss * self.cameraMatrix[i][j]

        self.viewMatrix = [
            [
                self.cameraMatrix[i][j]
                for j in range(len(self.cameraMatrix), len(self.cameraMatrix[0]))
            ]
            for i in range(len(self.cameraMatrix))
        ]

    def glProMatrix(self, n=0.1, f=1000, fov=60):
        aR = self.viewPortWidth / self.viewPortHeight
        tan1 = math.tan((fov * math.pi / 180) / 2) * n
        rt = aR * tan1

        divNRT = n / rt

        self.projectionMatrix = [
            [divNRT, 0, 0, 0],
            [0, n / tan1, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]

    # (10 puntos) Deben crear una función glClearColor(r, g, b) con la que se pueda cambiar el color con el que funciona glClear().
    # Los parámetros deben ser números en el rango de 0 a 1.
    def glClearColor(self, r, g, b):
        self.clearColor = color(r, g, b)

    # (15 puntos) Deben crear una función glColor(r, g, b) con la que se pueda cambiar el color con el que funciona glVertex().
    # Los parámetros deben ser números en el rango de 0 a 1
    def glColor(self, r, g, b):
        self.currentColor = color(r, g, b)

    # (20 puntos) Deben crear una función glClear() que llene el mapa de bits con un solo color
    def glClear(self):
        # Pixels
        self.pixels = [
            [self.clearColor for y in range(self.height)] for x in range(self.width)
        ]
        # Z buffer
        self.zbuffer = [
            [float("inf") for y in range(self.height)] for x in range(self.width)
        ]

    def glClearViewPort(self, cl=None):
        yValue1 = self.viewPortX + self.viewPortWidth
        yValue2 = self.viewPortY + self.viewPortHeight
        for x in range(self.viewPortX, yValue1):
            for y in range(self.viewPortY, yValue2):
                # Draw pixel viewport
                self.glPoint(x, y, cl)

    def glPoint(self, x, y, cl=None):
        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[x][y] = cl or self.currentColor

    def glPointViewport(self, x, y, cl=None):
        if x < -1 or x > 1 or y < -1 or y > 1:
            return
        tempX = x + 1
        tempY = y + 1
        tempVw = self.viewPortWidth / 2
        tempVh = self.viewPortHeight / 2
        finalX = tempX * tempVw + self.viewPortX
        finalY = tempY * tempVh + self.viewPortY

        finalX = int(finalX)
        finalY = int(finalY)

        self.glPoint(finalX, finalY, cl)

    def glRotMatrix(self, p=0, y=0, r=0):
        pi180 = math.pi / 180
        p *= pi180
        y *= pi180
        r *= pi180

        pMatrix = [
            [1, 0, 0, 0],
            [0, math.cos(p), -math.sin(p), 0],
            [0, math.sin(p), math.cos(p), 0],
            [0, 0, 0, 1],
        ]
        yMatrix = [
            [math.cos(y), 0, math.sin(y), 0],
            [0, 1, 0, 0],
            [-math.sin(y), 0, math.cos(y), 0],
            [0, 0, 0, 1],
        ]
        rMatrix = [
            [math.cos(r), -math.sin(r), 0, 0],
            [math.sin(r), math.cos(r), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        resultado = matrixMultiplications([pMatrix, yMatrix, rMatrix])
        return resultado

    def glLine(self, x0: V2, x1: V2, cl=None):
        tempX0 = int(x0.x)
        tempX1 = int(x1.x)
        tempY0 = int(x0.y)
        tempY1 = int(x1.y)

        if tempX0 == tempX1 and tempY0 == tempY1:
            self.glPoint(tempX0, tempY0, cl)
            return

        dy = tempY1 - tempY0
        dx = tempX1 - tempX0

        dy = abs(dy)
        dx = abs(dx)

        pendiente = dy > dx

        if pendiente == True:
            tempX0, tempY0 = tempY0, tempX0
            tempX1, tempY1 = tempY1, tempX1

        if pendiente == False:
            tempX0, tempY0 = tempX0, tempY0
            tempX1, tempY1 = tempX1, tempY1

        x1Bx0 = tempX1 < tempX0

        if x1Bx0 == True:
            tempX0, tempX1 = tempX1, tempX0
            tempY0, tempY1 = tempY1, tempY0

        if x1Bx0 == False:
            tempX0, tempX1 = tempX0, tempX1
            tempY0, tempY1 = tempY0, tempY1

        dx = tempX1 - tempX0
        dy = tempY1 - tempY0

        dy = abs(dy)
        dx = abs(dx)

        offset = 0
        limite = 0.5
        ecuacionRecta = dy / dx
        finalY = tempY0

        x1MasUno = tempX1 + 1

        for i in range(tempX0, x1MasUno):
            if pendiente == True:
                self.glPoint(finalY, i, cl)
            else:
                self.glPoint(i, finalY, cl)
            offset += ecuacionRecta
            if offset >= limite:
                finalY = finalY + 1 if tempY0 < tempY1 else finalY - 1
                limite += 1

    def glPolygon(self, polygon, cl=None) -> None:
        for i in range(len(polygon)):
            tempVar = i + 1
            moduleOp = tempVar % len(polygon)
            self.glLine(polygon[i], polygon[moduleOp], cl)
        self.glFillPolygon(polygon, cl, cl)

    def glFillPolygon(self, polygon, cl=None, color2=None) -> None:
        minX = min(polygon, key=lambda v: v.x).x
        maxX = max(polygon, key=lambda v: v.x).x
        minY = min(polygon, key=lambda v: v.y).y
        maxY = max(polygon, key=lambda v: v.y).y

        verifyMax = maxX > minX
        if verifyMax == True:
            minX = int(minX)
            maxX = int(maxX)
            maxMinX = maxX + minX
            maxMinY = maxY + minY
            mainX = maxMinX / 2
            mainY = maxMinY / 2
            mainX = int(mainX)
            mainY = int(mainY)

        def glColoring(x, y, cl, color2, option) -> None:
            YPlus = y + 1
            XPlus = x + 1
            XMinus = x - 1
            YMinus = y - 1
            if option == 1:
                if self.pixels[x][y] != cl:
                    self.glPoint(x, y, color2)
                    glColoring(x, YPlus, cl, color2, 1)
                    glColoring(XMinus, y, cl, color2, 1)
                    glColoring(XPlus, y, cl, color2, 1)
            if option == 2:
                if self.pixels[x][y] != cl:
                    self.glPoint(x, y, color2)
                    glColoring(x, YMinus, cl, color2, 2)
                    glColoring(XMinus, y, cl, color2, 2)
                    glColoring(XPlus, y, cl, color2, 2)

        glColoring(mainX, mainY, cl, color2, 1)
        self.glPoint(mainX, mainY, color(0, 0, 0))
        glColoring(mainX, mainY, cl, color2, 2)
        self.glPoint(mainX, mainY, color(0, 0, 0))

    def glModel(
        self,
        filename,
        translation=V3(0, 0, 0),
        rotation=V3(0, 0, 0),
        scalationFactor=V3(1, 1, 1),
    ):

        modelImport = ObjectOpener(filename)
        mMatrix = self.glCreateMatrix(translation, rotation, scalationFactor)
        rMatrix = self.glRotMatrix(rotation[0], rotation[1], rotation[2])

        for i in modelImport.faces:
            count = len(i)

            # VERTICES
            f1Temp = i[0][0] - 1
            f2Temp = i[1][0] - 1
            f3Temp = i[2][0] - 1

            # TEXTURES
            f1Text = i[0][1] - 1
            f2Text = i[1][1] - 1
            f3Text = i[2][1] - 1

            # NORMALS
            f1TempTX = i[0][2] - 1
            f2TempTX = i[1][2] - 1
            f3TempTX = i[2][2] - 1

            # Vertices
            f1 = modelImport.vertices[f1Temp]
            f2 = modelImport.vertices[f2Temp]
            f3 = modelImport.vertices[f3Temp]

            f1Transformed = self.glTransform(f1, mMatrix)
            f2Transformed = self.glTransform(f2, mMatrix)
            f3Transformed = self.glTransform(f3, mMatrix)

            A = self.glCam(f1Transformed)
            B = self.glCam(f2Transformed)
            C = self.glCam(f3Transformed)

            f1Textura = modelImport.texcoords[f1Text]
            f2Textura = modelImport.texcoords[f2Text]
            f3Textura = modelImport.texcoords[f3Text]

            f1Normal = modelImport.normals[f1TempTX]
            f2Normal = modelImport.normals[f2TempTX]
            f3Normal = modelImport.normals[f3TempTX]

            f1Normal = self.glDirT(f1Normal, rMatrix)
            f2Normal = self.glDirT(f2Normal, rMatrix)
            f3Normal = self.glDirT(f3Normal, rMatrix)

            self.glTriangleBary(
                A,
                B,
                C,
                ver=(f1Transformed, f2Transformed, f3Transformed),
                texcoords1=(f1Textura, f2Textura, f3Textura),
                normals=(f1Normal, f2Normal, f3Normal),
            )

            if count == 4:
                f4Temp = i[3][0] - 1
                f4Temp1 = i[3][1] - 1
                f4TempN = i[3][2] - 1
                f41 = modelImport.vertices[f4Temp]
                f4Transformed1 = self.glTransform(f41, mMatrix)
                D = self.glCam(f4Transformed1)
                f4Textura = modelImport.texcoords[f4Temp1]
                f4Normal = modelImport.normals[f4TempN]
                f4Normal = self.glDirT(f4Normal, rMatrix)

                self.glTriangleBary(
                    A,
                    C,
                    D,
                    ver=(f1Transformed, f3Transformed, f4Transformed1),
                    texcoords1=(f1Textura, f3Textura, f4Textura),
                    normals=(f1Normal, f3Normal, f4Normal),
                )

    def glLookAt(self, eye, camPos=V3(0, 0, 0)):
        Siguiente = [camPos.x - eye.x, camPos.y - eye.y, camPos.z - eye.z]
        siguiente1 = 0
        for i in Siguiente:
            siguiente1 += i**2
        resultadoSiguiente = pow(siguiente1, 0.5)
        Siguiente = [
            (i / resultadoSiguiente) for i in [Siguiente[0], Siguiente[1], Siguiente[2]]
        ]

        list1 = [0, 1, 0]
        if len(list1) != 3 and len(Siguiente) != 3:
            return None
        i = (list1[1] * Siguiente[2]) - (list1[2] * Siguiente[1])
        j = (list1[0] * Siguiente[2]) - (list1[2] * Siguiente[0])
        k = (list1[0] * Siguiente[1]) - (list1[1] * Siguiente[0])
        Derecho = [i, -j, k]

        derecho1 = 0
        for i in Derecho:
            derecho1 += i**2
        resultadoDerecho = pow(derecho1, 0.5)

        Derecho = [(i / resultadoDerecho) for i in Derecho]

        if len(Siguiente) != 3 and len(Derecho) != 3:
            return None
        i2 = (Siguiente[1] * Derecho[2]) - (Siguiente[2] * Derecho[1])
        j2 = (Siguiente[0] * Derecho[2]) - (Siguiente[2] * Derecho[0])
        k2 = (Siguiente[0] * Derecho[1]) - (Siguiente[1] * Derecho[0])
        Arriba = [i2, -j2, k2]

        arrib1 = 0
        for i in Arriba:
            arrib1 += i**2
        resultadoArriba = pow(arrib1, 0.5)

        Arriba = [(i / resultadoArriba) for i in Arriba]
        self.cameraMatrix = [
            [Derecho[0], Arriba[0], Siguiente[0], camPos[0]],
            [Derecho[1], Arriba[1], Siguiente[1], camPos[1]],
            [Derecho[2], Arriba[2], Siguiente[2], camPos[2]],
            [0, 0, 0, 1],
        ]

        tam = len(self.cameraMatrix)
        if tam != len(self.cameraMatrix[0]):
            return None

        identidadM = [
            [1 if x == y else 0 for y in range(0, tam)] for x in range(0, tam)
        ]
        for i in range(tam):
            self.cameraMatrix[i].extend(identidadM[i])

        for i in range(tam):
            verMat = self.cameraMatrix[i][i]
            if verMat == 0:
                return None

            for j in range(i, 2 * tam):
                self.cameraMatrix[i][j] /= verMat
            for z in range(tam):
                if z == i or self.cameraMatrix[z][i] == 0:
                    continue
                factor = self.cameraMatrix[z][i]
                for j in range(i, 2 * tam):
                    self.cameraMatrix[z][j] -= factor * self.cameraMatrix[i][j]

        self.viewMatrix = [
            [self.cameraMatrix[i][j] for j in range(tam, len(self.cameraMatrix[0]))]
            for i in range(tam)
        ]

    def glTransform(self, vertex, matrix):
        tempVertex = V4(vertex[0], vertex[1], vertex[2], 1)
        final = matrixMultiplication4x4(matrix, tempVertex)

        f0 = final[0]
        f3 = final[3]
        f1 = final[1]
        f2 = final[2]

        f03 = f0 / f3
        f13 = f1 / f3
        f23 = f2 / f3

        return V3(f03, f13, f23)

    def glDirT(self, dV, rM):

        resultado = V3(
            matrixMultiplication4x4(rM, V4(dV[0], dV[1], dV[2], 0))[0],
            matrixMultiplication4x4(rM, V4(dV[0], dV[1], dV[2], 0))[1],
            matrixMultiplication4x4(rM, V4(dV[0], dV[1], dV[2], 0))[2],
        )
        return resultado

    def glCam(self, v):
        v1 = V4(v[0], v[1], v[2], 1)

        vt = matrixMultiplication4x4(
            matrixMultiplications(
                [self.viewPortM, self.projectionMatrix, self.viewMatrix]
            ),
            v1,
        )

        return V3(vt[0] / vt[3], vt[1] / vt[3], vt[2] / vt[3])

    def glTriangle(self, A: V3, B: V3, C: V3, co=None):
        biggerAB_Y = A.y < B.y
        biggerAC_Y = A.y < C.y
        biggerBC_Y = B.y < C.y

        if biggerAB_Y == True:
            A, B = B, A
        if biggerAB_Y == False:
            A, B = A, B

        if biggerAC_Y == True:
            A, C = C, A
        if biggerAC_Y == False:
            A, C = A, C

        if biggerBC_Y == True:
            B, C = C, B
        if biggerBC_Y == False:
            B, C = B, C

        self.glLine(A, B, co)
        self.glLine(B, C, co)
        self.glLine(C, A, co)

    def glCreateMatrix(
        self, translation=V3(0, 0, 0), rotation=V3(0, 0, 0), scalationFactor=V3(1, 1, 1)
    ):

        tr1Row = [1, 0, 0, translation.x]
        tr2Row = [0, 1, 0, translation.y]
        tr3Row = [0, 0, 1, translation.z]
        tr4Row = [0, 0, 0, 1]

        traslateMatrix = [tr1Row, tr2Row, tr3Row, tr4Row]

        rot = self.glRotMatrix(rotation.x, rotation.y, rotation.z)

        xRow = [scalationFactor.x, 0, 0, 0]
        yRow = [0, scalationFactor.y, 0, 0]
        zRow = [0, 0, scalationFactor.z, 0]
        lastRow = [0, 0, 0, 1]
        rowlist = [xRow, yRow, zRow, lastRow]

        matrixListing = [traslateMatrix, rot, rowlist]

        matrixMultiplicationActual = matrixMultiplications(matrixListing)

        return matrixMultiplicationActual

    def glTriangleBary(self, A, B, C, ver=(), texcoords1=(), normals=(), color1=None):

        minX = min(A.x, B.x, C.x)
        minX = round(minX)

        minY = min(A.y, B.y, C.y)
        minY = round(minY)

        maxX = max(A.x, B.x, C.x)
        maxX = round(maxX)

        maxY = max(A.y, B.y, C.y)
        maxY = round(maxY)

        firstResta = [ver[1].x - ver[0].x, ver[1].y - ver[0].y, ver[1].z - ver[0].z]
        secondResta = [ver[2].x - ver[0].x, ver[2].y - ver[0].y, ver[2].z - ver[0].z]

        i = (firstResta[1] * secondResta[2]) - (firstResta[2] * secondResta[1])
        j = (firstResta[0] * secondResta[2]) - (firstResta[2] * secondResta[0])
        k = (firstResta[0] * secondResta[1]) - (firstResta[1] * secondResta[0])
        triangleNormal = [i, -j, k]

        def nD(infoXD):
            varRel = 0
            for i in infoXD:
                varRel += i**2
            return pow(varRel, 0.5)

        triangleNormal = [(i / nD(triangleNormal)) for i in triangleNormal]

        maxPlusOne = maxX + 1
        maxYPlusOne = maxY + 1
        for x in range(minX, maxPlusOne):
            for y in range(minY, maxYPlusOne):
                P = V2(x, y)
                u, v, w = baryCoords(A, B, C, P)
                if 0 <= u and 0 <= v and 0 <= w:
                    z = A.z * u + B.z * v + C.z * w
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if z < self.zbuffer[x][y] and -1 <= z <= 1:
                            self.zbuffer[x][y] = z
                            if self.shaderUsed:
                                r, g, b = self.shaderUsed(
                                    self,
                                    baryCoords=(u, v, w),
                                    colorU=color1 or self.currentColor,
                                    textureCoords=texcoords1,
                                    normals=normals,
                                    triangleNormal=triangleNormal,
                                )

                                colorUsed = color(r, g, b)
                                self.glPoint(x, y, colorUsed)
                            else:
                                self.glPoint(x, y, color1)

    def glFinish(self, filename):
        # constantes y calculos
        headerSize = headerInfo1 + (self.width * self.height * 3)
        widthD = dword(self.width)
        heightD = dword(self.height)
        whD = dword(self.width * self.height * 3)

        with open(filename, "wb") as f:
            f.write(char("B"))
            f.write(char("M"))
            f.write(dword(headerSize))
            f.write(Wcero)
            f.write(Wcero)
            f.write(dword(headerInfo1))
            f.write(w40)
            f.write(widthD)
            f.write(heightD)
            f.write(firstW)
            f.write(word24)
            f.write(Dcero)
            f.write(whD)
            f.write(Dcero)
            f.write(Dcero)
            f.write(Dcero)
            f.write(Dcero)

            for y in range(0, self.height):
                for x in range(0, self.width):
                    f.write(self.pixels[x][y])
            f.close()
