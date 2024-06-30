import math

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from Game.RC1Game import RC1Game
from Game.Game import Vector3
import numpy as np
from math import acos, degrees, sqrt


GLOBAL_FORWARD = Vector3(0, 0, 1)
GLOBAL_RIGHT = Vector3(1, 0, 0)
GLOBAL_UP = Vector3(0, 1, 0)


def extract_vector(vector):
    return np.array([vector.x, vector.y, vector.z])


def get_collisions_with_normals(game):
    collisions = []
    normals_x = []
    normals_y = []
    normals_z = []
    classes = []

    for i in range(8):
        for j in range(8):
            offset = 4 * (i * 8 + j)

            collision_address = game.collisions_address + offset
            collision_value = game.process.read_float(collision_address)

            normal_x_address = game.collisions_normals_x + offset
            normal_x = game.process.read_float(normal_x_address)

            normal_y_address = game.collisions_normals_y + offset
            normal_y = game.process.read_float(normal_y_address)

            normal_z_address = game.collisions_normals_z + offset
            normal_z = game.process.read_float(normal_z_address)

            class_address = game.collisions_class_address + offset
            class_value = game.process.read_int(class_address)

            if class_value > 5000:
                class_value = -1

            collisions.append(collision_value)
            normals_x.append(normal_x)
            normals_y.append(normal_y)
            normals_z.append(normal_z)
            classes.append(class_value)

    return np.array(collisions), np.array(normals_x), np.array(normals_y), np.array(normals_z), np.array(classes)


def class_to_color(class_value):
    if class_value < 0:
        return [1.0, 1.0, 1.0]  # White for no collision
    norm_class = class_value / 4096.0
    if norm_class < 1 / 6:
        r, g, b = 1.0, norm_class * 6, 0.0
    elif norm_class < 2 / 6:
        r, g, b = (2 - norm_class * 6), 1.0, 0.0
    elif norm_class < 3 / 6:
        r, g, b = 0.0, 1.0, (norm_class * 6 - 2)
    elif norm_class < 4 / 6:
        r, g, b = 0.0, (4 - norm_class * 6), 1.0
    elif norm_class < 5 / 6:
        r, g, b = (norm_class * 6 - 4), 0.0, 1.0
    else:
        r, g, b = 1.0, 0.0, (6 - norm_class * 6)
    return [r, g, b]


def draw_plane(color):
    size = 1.0
    glColor3fv(color)
    glBegin(GL_QUADS)
    glVertex3f(-size, -size, 0)
    glVertex3f(size, -size, 0)
    glVertex3f(size, size, 0)
    glVertex3f(-size, size, 0)
    glEnd()


def render_scene(game, camera_position, collisions, normals_x, normals_y, normals_z, classes):
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            distance = collisions[idx]

            if distance < 0 or np.isnan(distance):  # Skip invalid distances
                continue

            normal = np.array([normals_x[idx], normals_z[idx], normals_y[idx]])
            if np.any(np.isnan(normal)) or np.all(normal == 0):  # Skip invalid normals
                continue

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            color = class_to_color(classes[idx])
            glPushMatrix()
            # Position relative to camera
            position = [camera_position[0] + j - 4, camera_position[1] - (i - 4), camera_position[2] - distance]
            print(f"Position: {position}, Normal: {normal}, Color: {color}")  # Debugging information
            glTranslatef(position[0], position[1], position[2])

            # Calculate the angle between the normal and the global forward vector
            dot_product = np.clip(np.dot(normal, GLOBAL_FORWARD.numpy()), -1.0, 1.0)
            angle = degrees(acos(dot_product))

            # Calculate the axis of rotation
            axis = np.cross(GLOBAL_FORWARD.numpy(), normal)
            if np.linalg.norm(axis) > 0:  # Check if the axis is not zero
                axis = axis / np.linalg.norm(axis)  # Normalize the axis
                glRotatef(angle, axis[0], axis[1], axis[2])

            draw_plane(color)
            glPopMatrix()


def render_test_plane(camera_position, camera_forward):
    glPushMatrix()
    # Place the plane directly in front of the camera based on the forward vector
    test_plane_position = camera_position + 5 * camera_forward
    print(f"Rendering test plane at position: {test_plane_position}")
    glTranslatef(test_plane_position[0], test_plane_position[1], test_plane_position[2])
    draw_plane([1.0, 0.0, 0.0])  # Red plane for visibility
    glPopMatrix()


def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)

    ambient_light = [0.2, 0.2, 0.2, 1.0]
    diffuse_light = [0.8, 0.8, 0.8, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)


def update_light_position():
    light_position = [0.0, 1.0, -5, 1.0]  # Light at the camera's position
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)  # Enable depth testing
    glDepthFunc(GL_LEQUAL)  # Set depth function
    glClearDepth(1.0)  # Clear depth buffer

    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)  # Increased far clipping plane

    # Disable face culling
    glDisable(GL_CULL_FACE)

    setup_lighting()

    game = RC1Game()
    game.open_process()
    game.set_should_render(True)

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        camera_position = game.get_camera_position()  # Assuming this returns a Vector3 with the camera's position
        camera_forward, camera_right, camera_up = game.get_camera_vectors()

        # Adjust coordinate system: Game's (X, Y, Z) to OpenGL's (X, Z, Y)
        camera_position = np.array([camera_position.x, camera_position.z, camera_position.y])
        camera_forward = np.array([camera_forward.x, camera_forward.z, camera_forward.y])
        camera_right = np.array([camera_right.x, camera_right.z, camera_right.y])
        camera_up = np.array([camera_up.x, camera_up.z, camera_up.y])

        # Print out the camera position and orientation
        print(f"Camera Position: {camera_position}")
        print(f"Camera Forward: {camera_forward}")
        print(f"Camera Right: {camera_right}")
        print(f"Camera Up: {camera_up}")

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_position[0], camera_position[1], camera_position[2],
                  camera_position[0] + camera_forward[0], camera_position[1] + camera_forward[1],
                  camera_position[2] + camera_forward[2],
                  camera_up[0], camera_up[1], camera_up[2])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        update_light_position()
        render_test_plane(camera_position, camera_forward)  # Render a test plane relative to the camera
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
