"""
Autor: Alejandro Gómez
Fecha de última modificación: 14/07/22

"""
from gl import *


# Librerías para el título
import time
import sys

from SoftwareRenderer import *


# Funciones para el menu:
def Opciones():
    menuOp = input(
        "1) Ver Todos \n2) Medium shot \n3) Low angle \n4) High angle \n5) Dutch angle \n6) Salir\n"
    )
    menuVer = verificarNum(menuOp)
    return menuVer


def verificarNum(input):
    try:
        val = float(input)
        return val
    except ValueError:
        try:
            val = int(input)
            return val
        except ValueError:
            print("¡Solamente números!")


# Mensaje de despedida
def bye():
    print("¡Gracias por usar el programa!!")


Bienvenida = "\n----- GL Library----\n"
procesando = "Procesando solicitud..."


def ImpresionTitulo(string):
    # Se imprime el título con efecto de typewriter
    for i in string:
        sys.stdout.write(i)
        sys.stdout.flush()
        time.sleep(0.02)


ImpresionTitulo(Bienvenida)
menu = True
while menu == True:
    opcion = Opciones()
    if opcion == 1:
        tituloArchivo = input(
            "Ingrese el nombre para el archivo (NO incluir extension .bpm): "
        )

        ImpresionTitulo(procesando)
        SR6(tituloArchivo, 1)
        SR6(tituloArchivo, 2)
        SR6(tituloArchivo, 3)
        SR6(tituloArchivo, 4)
        print("\n\n¡Imagen generada!\n")

    if opcion == 2:
        tituloArchivo = input(
            "Ingrese el nombre para el archivo (NO incluir extension .bpm): "
        )

        ImpresionTitulo(procesando)
        SR6(tituloArchivo, 1)
        print("\n\n¡Imagen generada!\n")

    if opcion == 3:
        tituloArchivo = input(
            "Ingrese el nombre para el archivo (NO incluir extension .bpm): "
        )

        ImpresionTitulo(procesando)
        SR6(tituloArchivo, 2)
        print("\n\n¡Imagen generada!\n")

    if opcion == 4:
        tituloArchivo = input(
            "Ingrese el nombre para el archivo (NO incluir extension .bpm): "
        )

        ImpresionTitulo(procesando)
        SR6(tituloArchivo, 3)
        print("\n\n¡Imagen generada!\n")

    if opcion == 5:
        tituloArchivo = input(
            "Ingrese el nombre para el archivo (NO incluir extension .bpm): "
        )

        ImpresionTitulo(procesando)
        SR6(tituloArchivo, 4)
        print("\n\n¡Imagen generada!\n")

    if opcion == 6:
        print("Gracias por utilizar este programa.")
        print("\n")
        menu = False
