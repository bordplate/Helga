import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from matplotlib.animation import FuncAnimation


# Pygame setup
bar_width = 80
spacing = 20

color_active = (0, 255, 0)
color_inactive = (255, 0, 0)

labels = ["LJoyX", "LJoyY", "RJoyX", "RJoyY", "R1", "Cross", "Square"]

# Pygame initialization for visualization
pygame.init()

bar_y = 0
bar_height = 220

face_y = bar_height
face_height = 300

raycast_y = bar_height + face_height
raycast_height = 400

screen_width, screen_height = 1000, bar_height + face_height + raycast_height

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Action Visualization")

font = pygame.font.Font(None, 24)
bigger_font = pygame.font.Font(None, 36)
# bigger_font = pygame.font.Font("/usr/share/fonts/TTF/Hack-Bold.ttf", 36)


state_values = deque(maxlen=400)


def draw_score_and_checkpoint(score, checkpoint, best=0):
    """
    Draw the score and checkpoint on the screen.
    """

    score_label = bigger_font.render(f"Score: %.2f" % score, True, (255, 255, 255))
    checkpoint_label = bigger_font.render(f"Checkpoint: {checkpoint}", True, (255, 255, 255))
    best_label = bigger_font.render(f"Best: {best}", True, (255, 255, 255))

    screen.blit(score_label, (420, 500))
    screen.blit(checkpoint_label, (420, 540))
    screen.blit(best_label, (420, 580))


def draw_state_value_face(state_value):
    """
    Draw a face that represents the state value, happy for positive and sad for negative from 1 - 5.
    There are 5 images in total, each representing a different state value.

    Picks an image from ./imgs/face_{state_value}.png
    """
    state_value = state_value.to('cpu').detach()
    state_values.append(state_value)

    max_value = max(abs(max(state_values)), abs(min(state_values)))

    state_value = (state_value / max_value) * 2

    screen.fill((0, 0, 255))

    if abs(state_value) > 5:
        face_value = 0
    else:
        state_value = max(-1.9, min(1.9, state_value))

        face_value = 3

        if state_value > 0.5:
            face_value = 4
        if state_value > 1.8:
            face_value = 5
        if state_value < -0.5:
            face_value = 2
        if state_value < -1.8:
            face_value = 1

    face_img = pygame.image.load(f"imgs/state-{face_value}.png")
    face_img = pygame.transform.scale(face_img, (face_height, face_height))

    screen.blit(face_img, (0, bar_height))

    # pygame.display.flip()


def render_raycast_data(raycast_data):
    """
    The first 64 values in raycast data is a flattened 2D 8x8 grid of floats between 0.0f and 64.0f.
    The last 64 values are data about which entity it has collided with. Values below 0 should be white. 0 and up should be colored.
    This function renders a heatmap, where collision (moby data below 0) is white and other entities are colored. The distance decides the opacity of each pixel.
    """
    mobys = raycast_data[-256:].copy()
    raycast_data = raycast_data[:-256].copy()

    mobys = mobys.reshape(16, 16)
    raycast_data = raycast_data.reshape(16, 16)

    # mobys = np.repeat(mobys, 4, axis=1)
    # raycast_data = np.repeat(raycast_data, 4, axis=1)

    # mobys[mobys < 0] = 4096  # Mark non-colliding areas as maximum value
    raycast_data[raycast_data < 0] = 64  # Mark values below 0

    raycast_data = raycast_data / 64
    raycast_data = np.clip(raycast_data, 0, 1)

    # Invert the grayscale mapping: 0 -> 255 (white), 1 -> 0 (black)
    # distance_data is (8, 8)
    distance_data = 1 - raycast_data

    # Create a color scale for mobys data
    colors = np.zeros((16, 16, 3), dtype=np.uint8)
    norm_mobys = mobys / 4096  # Normalize mobys data to range 0-1

    # Full color spectrum interpolation from red to white
    for i in range(16):
        for j in range(16):
            (r, g, b) = (255, 255, 255)

            if norm_mobys[i, j] >= 0:
                value = norm_mobys[i, j]

                if value < 1/6:
                    r, g, b = 255, int(255 * 6 * value), 0
                elif value < 2/6:
                    r, g, b = int(255 * (2 - 6 * value)), 255, 0
                elif value < 3/6:
                    r, g, b = 0, 255, int(255 * (6 * value - 2))
                elif value < 4/6:
                    r, g, b = 0, int(255 * (4 - 6 * value)), 255
                elif value < 5/6:
                    r, g, b = int(255 * (6 * value - 4)), 0, 255
                else:
                    r, g, b = 255, 0, int(255 * (6 - 6 * value))
            else:
                value = mobys[i, j]

                if value == -33:  # Ground
                    (r, g, b) = (255, 255, 128)
                elif value == -12:  # Walls?
                    (r, g, b) = (200, 200, 0)
                elif value == -14:  # More walls?
                    (r, g, b) = (200, 200, 128)
                elif value == -3:  # Lava
                    (r, g, b) = (255, 0, 0)
                elif value == -10 or value == -11:  # Different walls?
                    (r, g, b) = (128, 128, 0)
                elif value == -128:
                    pass
                else:
                    print(f"Unknown value: {value}")

            # Adjust brightness based on distance data
            brightness = distance_data[i, j]
            colors[i, j] = [r * brightness, g * brightness, b * brightness]

    # mask = mobys < 0
    # grayscale_rgb = np.stack([grayscale_data] * 3, axis=-1) * 255
    # colors[mask] = grayscale_rgb[mask]

    # Flip and rotate as needed
    colors = np.flip(colors, axis=1)
    colors = np.rot90(colors)

    # Scale up for better visibility
    colors = np.repeat(colors, 50//2, axis=0)
    colors = np.repeat(colors, 50//2, axis=1)

    # Create a Pygame surface
    raycast_surface = pygame.Surface(colors.shape[:2])
    pygame.surfarray.blit_array(raycast_surface, colors)

    screen.blit(raycast_surface, (0, bar_height + face_height))


def draw_bars(actions, state_value, progress):
    for i, action in enumerate(actions):
        # Clip action to -1 and 1
        action = max(-1, min(1, action))

        color = color_active if action > 0.5 else color_inactive

        if i < 4:
            color = color_active if abs(action) > 0.25 else color_inactive

        height = int(abs(action) * 90)  # Scale action value to height
        bar_x = i * (bar_width + spacing) + 50
        bar_y = bar_height - 40 - height

        if action < 0:
            bar_y += height

        pygame.draw.rect(screen, color, pygame.Rect(bar_x, bar_y - 90, bar_width, height))

        label = font.render(labels[i], True, (255, 255, 255))
        label_pos_x = bar_x + (bar_width - label.get_width()) // 2  # Calculate x position to center the label
        screen.blit(label, (label_pos_x, bar_height - 40))

    # Draw state_value bar and label, (state_value is between -1 and 1)
    state_value = max(-2, min(2, state_value)) / 2

    _state_value = (state_value + 1) / 2
    _state_value = max(0, min(1, _state_value))
    height = int(abs(state_value) * 90)
    bar_x = 7 * (bar_width + spacing) + 50
    bar_y = bar_height - 40 - height

    # Draw a progress bar at the bottom of the screen
    progress_bar_width = int(progress * screen_width)
    pygame.draw.rect(screen, (255, 0, 0xdb), pygame.Rect(0, bar_height - 10, progress_bar_width, 10))

    pygame.display.flip()

    # Pygame event handling
    for event in pygame.event.get():
        pass
