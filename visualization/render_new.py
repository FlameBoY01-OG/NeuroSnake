# visualization/render_new.py
"""
Research-grade Neural Network Visualization for NeuroSnake.
Features live animated activations, decision path highlighting, and Dueling DQN architecture display.
"""
import pygame
import sys
import time
import numpy as np
from env.snake_env import SnakeEnv
from model.agent import DQNAgent
from visualization.activations import ActivationVisualizer, compute_neuron_positions, COLORS

# ============ CONFIG ============
CELL_SIZE = 20
FPS = 10
NET_HEIGHT = 450
PANEL_Q_WIDTH = 280
PADDING = 15
MODEL_PATH = "model_checkpoints/policy_final.pth"

# Display settings
MAX_INPUT_NEURONS = 32  # All input features visible
MAX_SHARED_NEURONS = 64  # Sample from shared layer
MAX_STREAM_NEURONS = 32  # Sample from value/advantage streams

BG_COLOR = COLORS['base']
PANEL_BG = COLORS['mantle']
GRID_COLOR = COLORS['surface0']
SNAKE_COLOR = COLORS['green']
HEAD_COLOR = COLORS['teal']
FOOD_COLOR = COLORS['peach']
TEXT_COLOR = COLORS['text']


def clamp_color(c):
    return tuple(max(0, min(255, int(v))) for v in c)


def draw_grid(screen, env, x_offset, y_offset):
    """Draw grid lines for the game board."""
    for x in range(env.width + 1):
        start = (x_offset + x * CELL_SIZE, y_offset)
        end = (x_offset + x * CELL_SIZE, y_offset + env.height * CELL_SIZE)
        pygame.draw.line(screen, GRID_COLOR, start, end, 1)
    for y in range(env.height + 1):
        start = (x_offset, y_offset + y * CELL_SIZE)
        end = (x_offset + env.width * CELL_SIZE, y_offset + y * CELL_SIZE)
        pygame.draw.line(screen, GRID_COLOR, start, end, 1)


def draw_snake(screen, env, x_offset, y_offset):
    """Draw the snake with gradient from head to tail."""
    for i, p in enumerate(env.snake):
        alpha = 1.0 - (i / len(env.snake)) * 0.6  # Fade toward tail
        if i == 0:
            color = HEAD_COLOR
        else:
            color = tuple(int(c * alpha) for c in SNAKE_COLOR)
        
        rect = pygame.Rect(
            x_offset + p.x * CELL_SIZE + 1,
            y_offset + p.y * CELL_SIZE + 1,
            CELL_SIZE - 2,
            CELL_SIZE - 2
        )
        pygame.draw.rect(screen, clamp_color(color), rect, border_radius=3)


def draw_food(screen, env, x_offset, y_offset):
    """Draw food with a pulsing glow effect."""
    if env.food is None:
        return
    center_x = x_offset + env.food.x * CELL_SIZE + CELL_SIZE // 2
    center_y = y_offset + env.food.y * CELL_SIZE + CELL_SIZE // 2
    radius = CELL_SIZE // 2 - 2
    
    # Glow effect
    pygame.draw.circle(screen, clamp_color(tuple(c // 2 for c in FOOD_COLOR)), (center_x, center_y), radius + 2)
    pygame.draw.circle(screen, clamp_color(FOOD_COLOR), (center_x, center_y), radius)


def draw_info_panel(screen, q_vals, chosen_action, score, fps_text, x, y, w, h):
    """Draw the side panel with Q-values and game info."""
    panel_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, clamp_color(PANEL_BG), panel_rect)
    pygame.draw.rect(screen, clamp_color(COLORS['surface0']), panel_rect, 2)

    title_font = pygame.font.Font(None, 28)
    mono_font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 18)

    # Title
    title = title_font.render("Q-Values", True, TEXT_COLOR)
    screen.blit(title, (x + 15, y + 15))

    # Q-value bars
    actions = ["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"]
    q_min = float(np.min(q_vals))
    q_max = float(np.max(q_vals))
    span = q_max - q_min if q_max != q_min else 1.0

    bar_x = x + 15
    bar_w = w - 30

    for i, (name, q) in enumerate(zip(actions, q_vals)):
        q_norm = (q - q_min) / span
        row_y = y + 60 + i * 55

        # Action label with Q-value
        color = COLORS['green'] if i == chosen_action else TEXT_COLOR
        label = mono_font.render(f"{name}: {q:+6.2f}", True, color)
        screen.blit(label, (bar_x, row_y))

        # Progress bar background
        bg_rect = pygame.Rect(bar_x, row_y + 28, bar_w, 14)
        pygame.draw.rect(screen, clamp_color(COLORS['surface0']), bg_rect)

        # Progress bar foreground
        fg_rect = pygame.Rect(bar_x, row_y + 28, int(bar_w * q_norm), 14)
        bar_color = COLORS['green'] if i == chosen_action else COLORS['blue']
        pygame.draw.rect(screen, clamp_color(bar_color), fg_rect)
        
        # Border
        pygame.draw.rect(screen, clamp_color(COLORS['overlay0']), bg_rect, 1)

    # Game stats
    stats_y = y + 60 + 4 * 55 + 30
    stats = [
        f"Score: {int(score)}",
        f"FPS: {fps_text}",
        f"Action: {actions[chosen_action].split()[1]}"
    ]
    
    for i, stat in enumerate(stats):
        text = small_font.render(stat, True, TEXT_COLOR)
        screen.blit(text, (bar_x, stats_y + i * 25))


def draw_dueling_network(screen, activations, weights, visualizer, x, y, w, h):
    """
    Draw the Dueling DQN architecture with animated activations.
    Layout: Input -> Shared Features -> [Value Stream + Advantage Stream] -> Q-Values
    """
    # Background
    pygame.draw.rect(screen, clamp_color(COLORS['crust']), pygame.Rect(x, y, w, h))
    
    # Extract activations from dueling architecture
    input_acts = activations['input']
    shared_acts = activations['shared']
    value_acts = activations['value_stream']
    adv_acts = activations['advantage_stream']
    q_vals = activations['q_values']
    selected_action = activations['selected_action']
    
    # Sample neurons for display (keep all inputs, sample hidden layers)
    def sample_neurons(acts, max_n):
        if len(acts) <= max_n:
            return acts, np.arange(len(acts))
        indices = np.linspace(0, len(acts) - 1, max_n, dtype=int)
        return acts[indices], indices
    
    shared_sampled, shared_indices = sample_neurons(shared_acts, MAX_SHARED_NEURONS)
    value_sampled, value_indices = sample_neurons(value_acts, MAX_STREAM_NEURONS)
    adv_sampled, adv_indices = sample_neurons(adv_acts, MAX_STREAM_NEURONS)
    
    # Define layer sizes for layout
    layer_sizes = [
        len(input_acts),
        len(shared_sampled),
        len(value_sampled),
        len(q_vals)
    ]
    
    # Compute positions - we'll manually adjust for dueling architecture
    # Layout: [Input] -> [Shared] -> [Value/Advantage] -> [Output]
    padding_x = 60
    padding_y = 30
    usable_w = w - 2 * padding_x
    usable_h = h - 2 * padding_y
    
    # Horizontal positions
    x_input = x + padding_x
    x_shared = x + padding_x + usable_w * 0.3
    x_streams = x + padding_x + usable_w * 0.65
    x_output = x + padding_x + usable_w
    
    # Vertical positions (centered for each layer)
    def get_layer_positions(n_neurons, center_x, top_y, height):
        if n_neurons == 1:
            return [(center_x, int(top_y + height / 2))]
        spacing = height / (n_neurons + 1)
        return [(center_x, int(top_y + spacing * (i + 1))) for i in range(n_neurons)]
    
    input_pos = get_layer_positions(len(input_acts), x_input, y + padding_y, usable_h)
    shared_pos = get_layer_positions(len(shared_sampled), x_shared, y + padding_y, usable_h)
    
    # Split streams vertically
    stream_height = usable_h * 0.45
    value_pos = get_layer_positions(len(value_sampled), x_streams, y + padding_y, stream_height)
    adv_pos = get_layer_positions(len(adv_sampled), x_streams, y + padding_y + usable_h - stream_height, stream_height)
    
    output_pos = get_layer_positions(len(q_vals), x_output, y + padding_y, usable_h)
    
    # Draw connections (edges) first (behind neurons)
    # Input -> Shared
    W_shared = weights['feature_layer'][0]  # (256, 32)
    visualizer.draw_connections(
        screen, W_shared[shared_indices, :], input_acts, shared_sampled,
        input_pos, shared_pos, selected_dst_idx=None, highlight_path=False
    )
    
    # Shared -> Value stream
    W_value = weights['value_stream'][0]  # (128, 256)
    visualizer.draw_connections(
        screen, W_value[value_indices, :][:, shared_indices], shared_sampled, value_sampled,
        shared_pos, value_pos, selected_dst_idx=None, highlight_path=False
    )
    
    # Shared -> Advantage stream
    W_adv = weights['advantage_stream'][0]  # (128, 256)
    visualizer.draw_connections(
        screen, W_adv[adv_indices, :][:, shared_indices], shared_sampled, adv_sampled,
        shared_pos, adv_pos, selected_dst_idx=None, highlight_path=False
    )
    
    # Streams -> Output (simplified - just show selected action path)
    # Value contributes equally to all actions
    # Advantages determine action preference
    W_adv_out = weights['advantage_stream'][1]  # (4, 128)
    visualizer.draw_connections(
        screen, W_adv_out[:, adv_indices], adv_sampled, q_vals,
        adv_pos, output_pos, selected_dst_idx=selected_action, highlight_path=True
    )
    
    # Draw neurons (on top of edges)
    node_radius = 5
    
    # Input layer - smaller nodes
    visualizer.draw_neuron_layer(screen, input_acts, input_pos, base_radius=3)
    
    # Shared features
    visualizer.draw_neuron_layer(screen, shared_sampled, shared_pos, base_radius=node_radius)
    
    # Value stream
    visualizer.draw_neuron_layer(screen, value_sampled, value_pos, base_radius=node_radius)
    
    # Advantage stream
    visualizer.draw_neuron_layer(screen, adv_sampled, adv_pos, base_radius=node_radius)
    
    # Output layer - highlight selected action
    action_labels = ["UP", "DOWN", "LEFT", "RIGHT"]
    visualizer.draw_neuron_layer(
        screen, q_vals, output_pos, base_radius=8,
        labels=action_labels, selected_indices=[selected_action]
    )
    
    # Draw labels for architecture components
    label_font = pygame.font.Font(None, 20)
    labels = [
        ("Input", (x_input, y + 10)),
        ("Shared", (x_shared - 20, y + 10)),
        ("Value", (x_streams - 20, y + padding_y - 15)),
        ("Advantage", (x_streams - 35, y + padding_y + usable_h - stream_height - 15)),
        ("Q-Values", (x_output - 30, y + 10)),
    ]
    for text, pos in labels:
        rendered = label_font.render(text, True, COLORS['subtext0'])
        screen.blit(rendered, pos)


def main():
    pygame.init()
    
    # Initialize environment
    env = SnakeEnv(width=20, height=20)
    state_dim = env.reset().shape[0]
    
    # Load agent
    print(f"Loading model: {MODEL_PATH}")
    print(f"State dimension: {state_dim}")
    
    agent = DQNAgent(input_dim=state_dim, output_dim=4)
    try:
        agent.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with random initialization...")
    
    # Setup display
    game_width = env.width * CELL_SIZE
    game_height = env.height * CELL_SIZE
    screen_width = game_width + PANEL_Q_WIDTH + 3 * PADDING
    screen_height = max(game_height, NET_HEIGHT) + 2 * PADDING + game_height + PADDING
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NeuroSnake - Research-Grade RL Visualization")
    clock = pygame.time.Clock()
    
    # Initialize visualizer
    visualizer = ActivationVisualizer()
    
    # Game state
    state = env.reset(randomize_start=True)
    running = True
    paused = False
    step_count = 0
    
    print("\n=== Controls ===")
    print("SPACE: Pause/Resume")
    print("R: Reset")
    print("ESC: Quit")
    print("================\n")
    
    while running:
        dt = clock.tick(FPS)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    state = env.reset(randomize_start=True)
                    step_count = 0
                    print("Reset environment")
        
        # Update animation
        visualizer.update_animation(dt / 1000.0)
        
        # Game logic (if not paused)
        if not paused:
            # Get action from agent
            action = agent.select_action(state, eps=0.0)  # Greedy (no exploration)
            
            # Step environment
            next_state, reward, done = env.step(action)
            state = next_state
            step_count += 1
            
            if done:
                print(f"Game Over! Score: {env.score}, Steps: {step_count}")
                time.sleep(1)
                state = env.reset(randomize_start=True)
                step_count = 0
        
        # Get network activations and Q-values
        activations, weights = agent.get_activations(state)
        q_values = activations['q_values']
        chosen_action = activations['selected_action']
        
        # === RENDERING ===
        screen.fill(BG_COLOR)
        
        # Game board (bottom)
        game_x = PADDING
        game_y = screen_height - game_height - PADDING
        draw_grid(screen, env, game_x, game_y)
        draw_snake(screen, env, game_x, game_y)
        draw_food(screen, env, game_x, game_y)
        
        # Neural network visualization (top)
        net_x = PADDING
        net_y = PADDING
        net_w = game_width
        net_h = NET_HEIGHT
        draw_dueling_network(screen, activations, weights, visualizer, net_x, net_y, net_w, net_h)
        
        # Info panel (right side)
        panel_x = game_width + 2 * PADDING
        panel_y = PADDING
        panel_w = PANEL_Q_WIDTH
        panel_h = screen_height - 2 * PADDING
        fps_display = f"{clock.get_fps():.0f}"
        draw_info_panel(screen, q_values, chosen_action, env.score, fps_display, panel_x, panel_y, panel_w, panel_h)
        
        pygame.display.flip()
    
    pygame.quit()
    print("Visualization closed.")


if __name__ == "__main__":
    main()
