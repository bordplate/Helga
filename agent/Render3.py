import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import ctypes
import struct
import math

from Game.RC1Game import RC1Game


# Assuming RC1Game and Vector3 classes are defined as provided earlier

def init_pygame():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

def render_plane(normal, distance, size=1.0):
    glPushMatrix()
    glTranslatef(normal[0] * distance, normal[1] * distance, normal[2] * distance)
    glBegin(GL_QUADS)
    glVertex3f(-size, -size, 0)
    glVertex3f(size, -size, 0)
    glVertex3f(size, size, 0)
    glVertex3f(-size, size, 0)
    glEnd()
    glPopMatrix()

def main():
    init_pygame()
    rc1_game = RC1Game()
    rc1_game.open_process()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Get collision data
        collisions, normals_x, normals_y, normals_z, classes = rc1_game.get_collisions_with_normals()

        # Debug: Print the first few collisions and normals
        print("Collisions:", collisions[:5])
        print("Normals X:", normals_x[:5])
        print("Normals Y:", normals_y[:5])
        print("Normals Z:", normals_z[:5])
        print("Classes:", classes[:5])

        # Render each plane
        for i in range(len(collisions)):
            if classes[i] != -1 and all(-1e6 < x < 1e6 for x in [normals_x[i], normals_y[i], normals_z[i]]):  # Only render if valid
                normal = (normals_x[i], normals_y[i], normals_z[i])
                distance = collisions[i]
                render_plane(normal, distance)

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
