import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from matplotlib.animation import FuncAnimation


# Pygame setup
screen_width, screen_height = 1000, 220

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

screen_width, screen_height = 1000, bar_height + face_height

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Action Visualization")

font = pygame.font.Font(None, 24)


def draw_state_value_face(state_value):
    """
    Draw a face that represents the state value, happy for positive and sad for negative from 1 - 5.
    There are 5 images in total, each representing a different state value.

    Picks an image from ./imgs/face_{state_value}.png
    """
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

    if state_value < 0:
        bar_y += height

        # Gradient red to white
        col = (int(255 * _state_value), int(255 * (1 - _state_value)), int(255 * (1 - _state_value)))
    else:
        # Gradient green to white
        col = (int(255 * (1 - _state_value)), int(255 * _state_value), int(255 * (1 - _state_value)))

    pygame.draw.rect(screen, col, pygame.Rect(bar_x, bar_y - 90, bar_width, height))

    label = font.render("State Value", True, (255, 255, 255))
    label_pos_x = bar_x + (bar_width - label.get_width()) // 2
    screen.blit(label, (label_pos_x, bar_height - 40))

    # Draw a progress bar at the bottom of the screen
    progress_bar_width = int(progress * screen_width)
    pygame.draw.rect(screen, (255, 0, 0xdb), pygame.Rect(0, bar_height - 10, progress_bar_width, 10))

    pygame.display.flip()

    # Pygame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
