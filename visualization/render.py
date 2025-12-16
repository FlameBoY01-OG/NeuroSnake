# visualization/render.py
import pygame
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

# ---------- CONFIG ----------
CELL_SIZE = 20
FPS = 14
NET_HEIGHT = 360
PANEL_Q_WIDTH = 260
PADDING = 10
MAX_NEURONS_DISPLAY = 128
TOP_EDGES = 8
NODE_RADIUS = 7
NODE_BORDER = 2
MODEL_PATH = "model_checkpoints/policy_final.pth"

# Catppuccin-ish palette
CAT = {
    "surface0":  (20, 20, 26),
    "mantle":    (26, 26, 34),
    "crust":     (30, 30, 40),
    "text":      (230, 230, 235),
    "green":     (166, 227, 161),
    "lavender":  (196, 196, 255),
    "mauve":     (198, 160, 246),
    "peach":     (250, 179, 135),
    "sky":       (137, 220, 235),
}
BG_COLOR = CAT["surface0"]
PANEL_BG = CAT["mantle"]
GRID_COLOR = (40, 40, 48)
SNAKE_COLOR = CAT["green"]
HEAD_COLOR = (200, 255, 190)
FOOD_COLOR = CAT["peach"]
TEXT = CAT["text"]

def clamp_color(c):
    return tuple(max(0, min(255, int(v))) for v in c)

def draw_grid(screen, env, x_offset, y_offset):
    for x in range(env.width):
        for y in range(env.height):
            rect = pygame.Rect(x_offset + x * CELL_SIZE, y_offset + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

def draw_snake(screen, env, x_offset, y_offset):
    for i, p in enumerate(env.snake):
        color = HEAD_COLOR if i == 0 else SNAKE_COLOR
        rect = pygame.Rect(x_offset + p.x * CELL_SIZE, y_offset + p.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, clamp_color(color), rect)

def draw_food(screen, env, x_offset, y_offset):
    if env.food is None:
        return
    rect = pygame.Rect(x_offset + env.food.x * CELL_SIZE, y_offset + env.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, clamp_color(FOOD_COLOR), rect)

def draw_q_panel(screen, q_vals, chosen_action, state, score, fps_text, x, y, w, h):
    WHITE = TEXT
    GREY = (140, 140, 150)
    GREEN = CAT["green"]
    BG = clamp_color(PANEL_BG)

    panel_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, BG, panel_rect)

    title_font = pygame.font.SysFont("consolas", 18, bold=True)
    mono_font = pygame.font.SysFont("consolas", 14)
    small_font = pygame.font.SysFont("consolas", 13)

    screen.blit(title_font.render("Q-values (policy)", True, WHITE), (x + 12, y + 8))
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    q_min = float(np.min(q_vals))
    q_max = float(np.max(q_vals))
    span = q_max - q_min if q_max != q_min else 1.0
    bar_x = x + 12
    bar_w = w - 28
    for i, (name, q) in enumerate(zip(actions, q_vals)):
        q_norm = (q - q_min) / span
        row_y = y + 40 + i * 34
        label = f"{name:<5}: {q:+6.2f}"
        color = GREEN if i == chosen_action else WHITE
        screen.blit(mono_font.render(label, True, color), (bar_x, row_y - 2))
        bg_rect = pygame.Rect(bar_x, row_y + 16, bar_w, 10)
        pygame.draw.rect(screen, (36, 36, 44), bg_rect)
        fg_rect = pygame.Rect(bar_x, row_y + 16, int(bar_w * q_norm), 10)
        bar_color = GREEN if i == chosen_action else GREY
        pygame.draw.rect(screen, clamp_color(bar_color), fg_rect)

    st_y = y + 40 + 4 * 34 + 12
    screen.blit(title_font.render("State (11 features)", True, WHITE), (x + 12, st_y))
    st_y += 28
    try:
        st = np.array(state, dtype=float).flatten()
        screen.blit(small_font.render(f"Danger: {st[0]:.0f} {st[1]:.0f} {st[2]:.0f}", True, WHITE), (x + 12, st_y))
        screen.blit(small_font.render(f"Dir: {st[3]:.0f} {st[4]:.0f} {st[5]:.0f} {st[6]:.0f}", True, WHITE), (x + 12, st_y + 16))
    except Exception:
        pass

    screen.blit(title_font.render(f"Score: {int(score)}", True, WHITE), (x + 12, st_y + 40))
    screen.blit(mono_font.render(f"FPS: {fps_text}", True, WHITE), (x + 12, st_y + 64))

def sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)

def draw_network_web_top(screen, agent: DQNAgent, state: np.ndarray, x, y, w, h):
    pygame.draw.rect(screen, clamp_color(CAT["crust"]), pygame.Rect(x, y, w, h))

    acts, weights = agent.get_activations(state)
    # acts is [input(11), h1(128), h2(128), output(4)]
    # weights is [W1, W2, W3]
    # sample neurons to keep panel readable
    layer_count = len(acts)
    # prepare sampled indices per layer
    sampled_idxs = []
    for i, a in enumerate(acts):
        sampled_idxs.append(sample_indices(a.shape[0], MAX_NEURONS_DISPLAY if i not in (0, layer_count-1) else a.shape[0]))

    # compute xs for layers across width
    L = layer_count
    layer_xs = [int(x + i * ((w - 2 * PADDING) / max(1, (L - 1))) + PADDING) for i in range(L)]
    layer_ys = {}
    for i, idxs in enumerate(sampled_idxs):
        n = len(idxs)
        if n <= 1:
            layer_ys[i] = [int(y + h // 2)]
        else:
            top = y + PADDING + 28
            bottom = y + h - PADDING - 28
            layer_ys[i] = list(np.linspace(top, bottom, n).astype(int))

    # helper color mapping
    eps = 1e-9
    # get sampled activations arrays
    sampled_acts = [acts[i][sampled_idxs[i]] for i in range(len(acts))]

    combined_max = max((np.max(np.abs(a)) if a.size else 0) for a in sampled_acts + [np.array([eps])])

    def node_color(act):
        v = min(1.0, abs(act) / (combined_max + eps))
        base = np.array(CAT["mantle"])
        highlight = np.array(CAT["lavender"])
        green = np.array(CAT["green"])
        if act >= 0:
            col = base * (1 - 0.75 * v) + (green * 0.45 + highlight * 0.55) * (0.75 * v)
        else:
            col = base * (1 - 0.75 * v) + (np.array(CAT["mauve"]) * 0.7 + highlight * 0.3) * (0.75 * v)
        col = col + 14 * v
        return clamp_color(col)

    def output_color(val):
        if val >= 0:
            a = min(1.0, val / (np.max(np.abs(acts[-1])) + eps))
            col = np.array(CAT["sky"]) * a + np.array(CAT["lavender"]) * (1 - a)
        else:
            a = min(1.0, abs(val) / (np.max(np.abs(acts[-1])) + eps))
            col = np.array(CAT["peach"]) * a + np.array(CAT["mauve"]) * (1 - a)
        return clamp_color(col)

    # draw edges: weights[i] connects layer i (from) to layer i+1 (to)
    for i, W in enumerate(weights):
        from_idxs = sampled_idxs[i]
        to_idxs = sampled_idxs[i+1]
        from_act = sampled_acts[i]
        # for each to-node, pick top edges from from-nodes
        for tj, to_full in enumerate(to_idxs):
            strengths = []
            for fi, from_full in enumerate(from_idxs):
                # weight matrix shape (to_full, from_full) but we sample by indices -> full mapping required
                try:
                    wv = abs(W[to_full, from_full])
                except Exception:
                    # If weight matrix is smaller (because we sampled incorrectly) fallback to using relative indices
                    wv = abs(W[min(to_full, W.shape[0]-1), min(from_full, W.shape[1]-1)])
                av = abs(from_act[fi])
                strengths.append(wv * av)
            strengths = np.array(strengths)
            if strengths.sum() == 0:
                continue
            top_k = min(TOP_EDGES, len(from_idxs))
            top_inds = np.argsort(-strengths)[:top_k]
            max_s = strengths[top_inds].max(initial=1e-12)
            tx = layer_xs[i+1]
            ty = layer_ys[i+1][tj]
            for idx in top_inds:
                fx = layer_xs[i]
                fy = layer_ys[i][idx]
                s = strengths[idx]
                norm_s = float(s / (max_s + 1e-12))
                col = (int(CAT["mauve"][0] * norm_s + 40 * (1 - norm_s)),
                       int(CAT["mauve"][1] * norm_s + 60 * (1 - norm_s)),
                       int(CAT["mauve"][2] * norm_s + 90 * (1 - norm_s)))
                width = max(1, int(1 + 3 * norm_s))
                try:
                    pygame.draw.aaline(screen, clamp_color(col), (fx, fy), (tx, ty))
                    if width > 1:
                        for off in range(1, width):
                            pygame.draw.aaline(screen, clamp_color(col), (fx, fy + off), (tx, ty + off))
                except Exception:
                    pass

    # draw nodes
    for i, idxs in enumerate(sampled_idxs):
        for j, _ in enumerate(idxs):
            nx = layer_xs[i]
            ny = layer_ys[i][j]
            act_val = sampled_acts[i][j]
            if i == len(sampled_idxs) - 1:
                col = output_color(act_val)
                pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER + 2)
                pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS + 2)
            else:
                col = node_color(act_val)
                pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER)
                pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS)

    # title
    font = pygame.font.SysFont("consolas", 18, bold=True)
    screen.blit(font.render("Policy Network — Live (Catppuccin)", True, TEXT), (x + 12, y + 8))

if __name__ == "__main__":
    pygame.init()
    env = SnakeEnv(width=20, height=20)
    GAME_W = env.width * CELL_SIZE
    GAME_H = env.height * CELL_SIZE

    SCREEN_W = max(GAME_W + PANEL_Q_WIDTH + 3 * PADDING, 1200)
    SCREEN_H = NET_HEIGHT + GAME_H + 3 * PADDING

    screen = pygame.display.set_mode((int(SCREEN_W), int(SCREEN_H)))
    pygame.display.set_caption("NeuroSnake — Big Net (Catppuccin)")
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        try:
            print("CUDA:", torch.cuda.get_device_name(device))
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    # create an agent with input dim pulled from env state
    state_dim = env.reset().shape[0]
    agent = DQNAgent(input_dim=state_dim, output_dim=4, device=device)
    try:
        agent.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Could not load model (continuing with random weights):", e)

    agent.policy.eval()

    state = env.reset()
    score = 0
    running = True
    while running:
        dt = clock.tick(FPS)
        fps_text = f"{clock.get_fps():.1f}"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            st_t = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0).to(device)
            q_t = agent.policy(st_t).squeeze(0).cpu().numpy()

        action = int(np.argmax(q_t))
        next_state, reward, done = env.step(action)
        if reward >= 9.0:
            score += 1

        screen.fill(clamp_color(BG_COLOR))

        # network panel (top)
        net_x = PADDING
        net_y = PADDING
        net_w = SCREEN_W - 2 * PADDING
        net_h = NET_HEIGHT
        draw_network_web_top(screen, agent, state, net_x, net_y, net_w, net_h)

        # centered game
        left_area_width = SCREEN_W - PANEL_Q_WIDTH - 3 * PADDING
        game_x = PADDING + int(max(0, (left_area_width - GAME_W) / 2))
        game_y = net_y + net_h + PADDING
        panel_x = game_x + GAME_W + PADDING
        panel_y = game_y
        panel_w = PANEL_Q_WIDTH
        panel_h = GAME_H

        draw_grid(screen, env, game_x, game_y)
        draw_snake(screen, env, game_x, game_y)
        draw_food(screen, env, game_x, game_y)

        draw_q_panel(screen, q_t, action, state, score, fps_text, panel_x, panel_y, panel_w, panel_h)

        pygame.display.flip()

        state = next_state
        if done:
            time.sleep(0.35)
            state = env.reset()
            score = 0

    pygame.quit()
    sys.exit()
