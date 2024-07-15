import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import psutil

from Game.RC1Game import RC1Game

def draw_axes():
    glBegin(GL_LINES)

    # X axis in red
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(2.0, 0.0, 0.0)

    # Y axis in green
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 2.0, 0.0)

    # Z axis in blue
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 2.0)

    glEnd()

def draw_points(points, normals):
    glBegin(GL_POINTS)
    for i, point in enumerate(points):
        color = (i / len(points), 1.0 - i / len(points), 0.5)
        glColor3f(*color)
        glVertex3fv(point)
    glEnd()

    # Draw planes oriented according to normals
    for i, (point, normal) in enumerate(zip(points, normals)):
        color = (i / len(points), 1.0 - i / len(points), 0.5)
        draw_plane(point, normal, color)

def draw_plane(center, normal, color, size=0.5):
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.cross(normal, [0, 1, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
    v = np.cross(normal, u)

    u = u / np.linalg.norm(u) * size
    v = v / np.linalg.norm(v) * size

    p1 = center + u + v
    p2 = center - u + v
    p3 = center - u - v
    p4 = center + u - v

    glBegin(GL_QUADS)
    glColor3f(*color)
    glVertex3fv(p1)
    glVertex3fv(p2)
    glVertex3fv(p3)
    glVertex3fv(p4)
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -20)  # Move the camera back to view from the origin

    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 0, 1))  # Light at the origin
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))   # White light

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)

    pid = 0

    # Find the rpcs3 process
    for proc in reversed(list(psutil.process_iter())):
        if proc.name() == "rpcs3":
            pid = proc.pid
            break

    game = RC1Game(pid=pid)
    game.open_process()
    game.set_should_render(True)

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        distances, classes, normals_x, normals_y, normals_z = game.get_collisions_with_normals()
        distances = np.array(distances).reshape(8, 8)

        # Calculate points based on distances in the collisions
        points = []

        for i in range(8):
            for j in range(8):
                x = (i - 4) * 2  # Adjust the grid position
                y = (j - 4) * 2  # Adjust the grid position
                z = distances[i, j]
                points.append([x, y, z])

        points = np.array(points)

        # Get the normals from the last 64*3 values in collisions
        normals = np.array([*normals_x, *normals_y, *normals_z]).reshape(64, 3)

        # Ensure normals are correctly normalized
        for idx, normal in enumerate(normals):
            if np.linalg.norm(normal) > 0:
                normals[idx] = normal / np.linalg.norm(normal)

        print("Points:", points)
        print("Normals:", normals)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes()  # Draw coordinate axes for reference
        draw_points(points, normals)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
