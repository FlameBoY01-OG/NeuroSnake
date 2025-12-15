# visualization/q_panel.py
import pygame

WHITE = (240, 240, 240)
GREEN = (0, 220, 0)
GRAY = (180, 180, 180)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

def draw_q_panel(screen, q_values, chosen_action, x, y):
    """
    Draws Q-values vertically
    """
    font = pygame.font.SysFont("consolas", 18)

    for i, (name, q) in enumerate(zip(ACTIONS, q_values)):
        color = GREEN if i == chosen_action else WHITE
        text = f"{name:<5}: {q:+6.2f}"
        surf = font.render(text, True, color)
        screen.blit(surf, (x, y + i * 24))
