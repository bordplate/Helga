import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from Game.RC1Game import RC1Game

def draw_points(points, normals):
    glBegin(GL_POINTS)
    for point in points:
        glVertex3fv(point)
    glEnd()

    # Draw planes pointing in the normal direction
    for point, normal in zip(points, normals):
        draw_plane(point, normal)

def draw_plane(center, normal, size=0.5):
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
    glNormal3fv(normal)  # Set the normal for lighting
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
    # glTranslatef(0.0, 0.0, -10)  # Move the camera back to view from the origin

    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 0, 1))  # Light at the origin
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))   # White light

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)

    game = RC1Game()
    game.open_process()

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        collisions = np.array(game.get_collisions_with_normals())
        distances = collisions[:64].reshape(8, 8)

        # Calculate points based on distances in the collisions
        # We know which direction we should place the point by getting the grid position of the collision
        points = []

        for i in range(8):
            for j in range(8):
                distance = distances[i, j]

                # Calculate position where maximum distance is 64.0. The raycasts are calculated like this:
                # Vec4 forward = camera_forward;
                # forward.x = camera_forward.z;
                # forward.y = camera_right.z;
                # forward.z = camera_up.z;
                # forward.w = 0;
                #
                # Vec4 left = Vec4();
                # left.x = camera_forward.x;
                # left.y = camera_right.x;
                # left.z = camera_up.x;
                # left.w = 0;
                #
                # Vec4 up = camera_up;
                # up.x = camera_forward.y;
                # up.y = camera_right.y;
                # up.z = camera_up.y;
                #
                # float ray_distance = 64.0f;
                # float ray_wide = 90.0f;
                #
                # float fov = 64.0f;
                # int rows = 8;
                # int cols = 8;
                #
                # for (int i = 0; i < rows; i++) {
                #   for (int j = 0; j < cols; j++) {
                #   Vec4 ray = Vec4();
                #   ray.x = camera_pos.x + ray_distance * forward.x + (ray_wide * (j - cols/ 2) /cols) * left.x + (ray_wide * (i - rows/ 2) /rows) * up.x;
                #   ray.y = camera_pos.y + ray_distance * forward.y + (ray_wide * (j - cols/ 2) /cols) * left.y + (ray_wide * (i - rows/ 2) /rows) * up.y;
                #   ray.z = camera_pos.z + ray_distance * forward.z + (ray_wide * (j - cols/ 2) /cols) * left.z + (ray_wide * (i - rows/ 2) /rows) * up.z;
                #   ray.w = 1;
                #   Vec4 ray_start = camera_pos;
                #   ray_start.x += 0.5f * forward.x;
                #   ray_start.y += 0.5f * forward.y;
                #   ray_start.z += 0.5f * forward.z;
                #
                #   // Apply oscillation
                #   ray.x += oscillation_offset_x * left.x + oscillation_offset_y * up.x;
                #   ray.y += oscillation_offset_x * left.y + oscillation_offset_y * up.y;
                #   ray.z += oscillation_offset_x * left.z + oscillation_offset_y * up.z;
                #
                #   int coll = coll_line(&ray_start, &ray, 0x24, nullptr, nullptr);

                # Calculate the position of the point based on the distance and the angles
                theta = np.radians(90 - 90 * i / 8)  # Polar angle
                phi = np.radians(90 * j / 8)  # Azimuthal angle

                x = distance * np.sin(theta) * np.cos(phi)
                y = distance * np.sin(theta) * np.sin(phi)
                z = distance * np.cos(theta)

                points.append([x, y, z])

        points = np.array(points)

        # Get the normals from the last 64*3 values in collisions
        normals = collisions[64:].reshape(64, 3)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_points(points, normals)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
