# visualization/render.py
"""
NeuroSnake visualization: big neural-network web on top + centered game + Q-panel below,
with a Catppuccin-inspired theme and brighter nodes.

Run:
    python -m visualization.render

Requirements: pygame, torch, numpy
"""
import pygame
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

# ---------------- CONFIG ----------------
CELL_SIZE = 20
FPS = 16

# Layout sizes
NET_HEIGHT = 360                  # big neural net panel at the top
PANEL_Q_WIDTH = 260               # right-side info panel below the net
PADDING = 10

# Neural graph settings (bigger / more visible)
MAX_NEURONS_DISPLAY = 128         # max neurons to display per hidden layer
TOP_EDGES = 8                     # edges drawn per target node
NODE_RADIUS = 7                   # increased for visibility
NODE_BORDER = 2                   # light border ring
MODEL_PATH = "model_checkpoints/policy_final.pth"
# ----------------------------------------

# --- Catppuccin (Mocha-inspired) palette (approximate RGB tuples) ---
CAT = {
    "rosewater": (245, 224, 220),
    "flamingo":  (242, 205, 205),
    "pink":      (245, 194, 231),
    "mauve":     (198, 160, 246),
    "peach":     (250, 179, 135),
    "yellow":    (249, 226, 175),
    "green":     (166, 227, 161),
    "teal":      (148, 226, 213),
    "sky":       (137, 220, 235),
    "blue":      (137, 180, 250),
    "lavender":  (196, 196, 255),
    "surface0":  (20, 20, 26),    # dark background
    "mantle":    (26, 26, 34),
    "crust":     (30, 30, 40),
    "text":      (230, 230, 235),
}
# fallback aliases
BG_COLOR = CAT["surface0"]
PANEL_BG = CAT["mantle"]
GRID_COLOR = (40, 40, 48)
SNAKE_COLOR = CAT["green"]
HEAD_COLOR = (200, 255, 190)
FOOD_COLOR = CAT["peach"]
TEXT = CAT["text"]

# Helper: clamp color
def clamp_color(c):
    return tuple(max(0, min(255, int(v))) for v in c)

# ------------- drawing helpers -------------
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
    screen.blit(title_font.render("State (excerpt)", True, WHITE), (x + 12, st_y))
    st_y += 28

    try:
        st = np.array(state, dtype=float).flatten()
        danger_up, danger_down, danger_left, danger_right = st[0:4]
        food_left, food_right, food_up, food_down = st[8:12]
        head_x, head_y = st[12:14]
        food_x, food_y = st[14:16]
        length_norm = st[16]
        screen.blit(small_font.render(f"Dangers U D L R: {int(danger_up)} {int(danger_down)} {int(danger_left)} {int(danger_right)}", True, WHITE), (x + 12, st_y)); st_y += 18
        screen.blit(small_font.render(f"Food L R U D: {int(food_left)} {int(food_right)} {int(food_up)} {int(food_down)}", True, WHITE), (x + 12, st_y)); st_y += 18
        screen.blit(small_font.render(f"Head pos: ({head_x:.2f},{head_y:.2f})", True, WHITE), (x + 12, st_y)); st_y += 18
        screen.blit(small_font.render(f"Food pos: ({food_x:.2f},{food_y:.2f})", True, WHITE), (x + 12, st_y)); st_y += 18
        screen.blit(small_font.render(f"Length norm: {length_norm:.3f}", True, WHITE), (x + 12, st_y)); st_y += 22
    except Exception:
        screen.blit(small_font.render("State: (invalid shape)", True, WHITE), (x + 12, st_y))
        st_y += 20

    screen.blit(title_font.render(f"Score: {int(score)}", True, WHITE), (x + 12, st_y)); st_y += 28
    screen.blit(mono_font.render(f"FPS: {fps_text}", True, WHITE), (x + 12, st_y))


def compute_forward_activations(policy_net, state_tensor):
    """
    Returns (input_act, hid1_act, hid2_act, out_act) as numpy 1D arrays on CPU
    """
    with torch.no_grad():
        l0 = policy_net.net[0](state_tensor)
        a1 = F.relu(l0)
        l1 = policy_net.net[2](a1)
        a2 = F.relu(l1)
        out = policy_net.net[4](a2)
        return (
            state_tensor.squeeze(0).cpu().numpy(),
            a1.squeeze(0).cpu().numpy(),
            a2.squeeze(0).cpu().numpy(),
            out.squeeze(0).cpu().numpy()
        )


def sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)


def draw_network_web_top(screen, agent, state, x, y, w, h):
    """
    Draw a large network web in the top rectangle (x,y,w,h)
    """
    # background
    pygame.draw.rect(screen, clamp_color(CAT["crust"]), pygame.Rect(x, y, w, h))

    # get activations and weights
    st_t = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0).to(agent.device)
    input_act, hid1_act, hid2_act, out_act = compute_forward_activations(agent.policy, st_t)

    # weights (cpu)
    w0 = agent.policy.net[0].weight.detach().cpu().numpy()   # (hid1, input)
    w1 = agent.policy.net[2].weight.detach().cpu().numpy()   # (hid2, hid1)
    w2 = agent.policy.net[4].weight.detach().cpu().numpy()   # (out, hid2)

    # sample indices
    in_idx = sample_indices(input_act.shape[0], input_act.shape[0])
    h1_idx = sample_indices(hid1_act.shape[0], MAX_NEURONS_DISPLAY)
    h2_idx = sample_indices(hid2_act.shape[0], MAX_NEURONS_DISPLAY)
    out_idx = np.arange(out_act.shape[0], dtype=int)

    # layout: 4 vertical layers across width
    layers = [("in", in_idx), ("h1", h1_idx), ("h2", h2_idx), ("out", out_idx)]
    L = len(layers)
    layer_xs = [int(x + i * ((w - 2 * PADDING) / (L - 1)) + PADDING) for i in range(L)]

    # compute y positions by layer (spread within available height)
    layer_ys = {}
    for name, idxs in layers:
        n = len(idxs)
        if n <= 1:
            layer_ys[name] = [int(y + h // 2)]
        else:
            top = y + PADDING + 28
            bottom = y + h - PADDING - 28
            layer_ys[name] = list(np.linspace(top, bottom, n).astype(int))

    # sampled activation arrays
    in_act_samp = input_act[in_idx]
    h1_act_samp = hid1_act[h1_idx]
    h2_act_samp = hid2_act[h2_idx]

    # colors mapping helpers (Catppuccin)
    eps = 1e-9
    combined_max = max(
        np.max(np.abs(in_act_samp)) if in_act_samp.size else 0,
        np.max(np.abs(h1_act_samp)) if h1_act_samp.size else 0,
        np.max(np.abs(h2_act_samp)) if h2_act_samp.size else 0,
        eps
    )

    # Brighter node color: blend toward 'lavender' or 'green' more strongly for high activation
    def node_color(act):
        v = min(1.0, abs(act) / (combined_max + eps))
        # boost brightness by weighting more toward lavender for visibility
        base = np.array(CAT["mantle"])
        highlight = np.array(CAT["lavender"])
        green = np.array(CAT["green"])
        # blend depending on sign/magnitude: if positive prefer green-hint, negative prefer mauve hint
        if act >= 0:
            col = base * (1 - 0.75 * v) + (green * 0.45 + highlight * 0.55) * (0.75 * v)
        else:
            col = base * (1 - 0.75 * v) + (np.array(CAT["mauve"]) * 0.7 + highlight * 0.3) * (0.75 * v)
        # brightening factor
        col = col + 14 * v
        return clamp_color(col)

    def output_color(val):
        # positive -> sky/green, negative -> peach/peach-red (bright)
        if val >= 0:
            a = min(1.0, val / (np.max(np.abs(out_act)) + eps))
            col = np.array(CAT["sky"]) * a + np.array(CAT["lavender"]) * (1 - a)
        else:
            a = min(1.0, abs(val) / (np.max(np.abs(out_act)) + eps))
            col = np.array(CAT["peach"]) * a + np.array(CAT["flamingo"]) * (1 - a)
        return clamp_color(col)

    # draw strongest edges (for readability)
    def draw_edges(weight_mat, from_idxs, to_idxs, from_act, from_name, to_name):
        # weight_mat shape (to_full, from_full)
        for ti, to_full in enumerate(to_idxs):
            # compute strengths for sampled from-nodes
            strengths = []
            for fi, from_full in enumerate(from_idxs):
                wv = abs(weight_mat[to_full, from_full])
                av = abs(from_act[fi])
                strengths.append(wv * av)
            strengths = np.array(strengths)
            if strengths.size == 0:
                continue
            top_k = min(TOP_EDGES, strengths.size)
            top_inds = np.argsort(-strengths)[:top_k]
            if top_inds.size == 0:
                continue
            max_s = strengths[top_inds].max(initial=1e-12)
            # positions of target
            tx = layer_xs[["in", "h1", "h2", "out"].index(to_name)]
            # find ty index in sampled order
            try:
                ty = layer_ys[to_name][int(np.where(to_idxs == to_full)[0][0])]
            except Exception:
                ty = layer_ys[to_name][0]
            for idx in top_inds:
                fx = layer_xs[["in", "h1", "h2", "out"].index(from_name)]
                fy = layer_ys[from_name][idx]
                s = strengths[idx]
                norm_s = float(s / (max_s + 1e-12))
                col = (
                    int(CAT["mauve"][0] * norm_s + 40 * (1 - norm_s)),
                    int(CAT["mauve"][1] * norm_s + 60 * (1 - norm_s)),
                    int(CAT["mauve"][2] * norm_s + 90 * (1 - norm_s)),
                )
                width = max(1, int(1 + 3 * norm_s))
                try:
                    pygame.draw.aaline(screen, clamp_color(col), (fx, fy), (tx, ty))
                    if width > 1:
                        for off in range(1, width):
                            pygame.draw.aaline(screen, clamp_color(col), (fx, fy + off), (tx, ty + off))
                except Exception:
                    pass

    # draw edges for input->h1, h1->h2, h2->out
    draw_edges(w0, in_idx, h1_idx, in_act_samp, "in", "h1")
    draw_edges(w1, h1_idx, h2_idx, h1_act_samp, "h1", "h2")
    draw_edges(w2, h2_idx, out_idx, h2_act_samp, "h2", "out")

    # draw nodes (as circles) with brighter Catppuccin colors + light border
    # input nodes
    for i, idx in enumerate(in_idx):
        nx = layer_xs[0]
        ny = layer_ys["in"][i]
        col = node_color(in_act_samp[i])
        # border
        pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER)
        pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS)

    # h1 nodes
    for i, idx in enumerate(h1_idx):
        nx = layer_xs[1]
        ny = layer_ys["h1"][i]
        col = node_color(h1_act_samp[i])
        pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER)
        pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS)

    # h2 nodes
    for i, idx in enumerate(h2_idx):
        nx = layer_xs[2]
        ny = layer_ys["h2"][i]
        col = node_color(h2_act_samp[i])
        pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER)
        pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS)

    # out nodes (bigger)
    for i, idx in enumerate(out_idx):
        nx = layer_xs[3]
        ny = layer_ys["out"][i]
        col = output_color(out_act[idx])
        pygame.draw.circle(screen, clamp_color(CAT["lavender"]), (nx, ny), NODE_RADIUS + NODE_BORDER + 2)
        pygame.draw.circle(screen, col, (nx, ny), NODE_RADIUS + 2)

    # title
    font = pygame.font.SysFont("consolas", 18, bold=True)
    # screen.blit(font.render("Policy Network — Big (Catppuccin theme, brighter nodes)", True, TEXT), (x + 12, y + 8))


# ----------------- MAIN -----------------
if __name__ == "__main__":
    pygame.init()

    env = SnakeEnv(width=20, height=20)
    GAME_W = env.width * CELL_SIZE
    GAME_H = env.height * CELL_SIZE

    # ensure a reasonably wide window
    SCREEN_W = max(GAME_W + PANEL_Q_WIDTH + 3 * PADDING, 1200)
    SCREEN_H = NET_HEIGHT + GAME_H + 3 * PADDING

    screen = pygame.display.set_mode((int(SCREEN_W), int(SCREEN_H)))
    pygame.display.set_caption("NeuroSnake — Big Net (Catppuccin, bright nodes)")
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        try:
            print("CUDA:", torch.cuda.get_device_name(device))
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    agent = DQNAgent(device=device)
    try:
        agent.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Warning: could not load model:", MODEL_PATH, " — running with random weights. Error:", e)

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

        # get q-values
        with torch.no_grad():
            st_t = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0).to(device)
            q_t = agent.policy(st_t).squeeze(0).cpu().numpy()

        action = int(np.argmax(q_t))
        next_state, reward, done = env.step(action)
        if reward >= 9.0:
            score += 1

        # Rendering order:
        # 1) top: big network web
        # 2) bottom-left: centered game
        # 3) bottom-right: q panel
        screen.fill(clamp_color(BG_COLOR))

        # network panel (top)
        net_x = PADDING
        net_y = PADDING
        net_w = SCREEN_W - 2 * PADDING
        net_h = NET_HEIGHT
        draw_network_web_top(screen, agent, state, net_x, net_y, net_w, net_h)

        # compute centered game x within left area (space left of Q-panel)
        left_area_width = SCREEN_W - PANEL_Q_WIDTH - 3 * PADDING
        game_x = PADDING + int(max(0, (left_area_width - GAME_W) / 2))
        game_y = net_y + net_h + PADDING

        # q-panel is placed to the right of the game area (preserve PADDING)
        panel_x = game_x + GAME_W + PADDING
        panel_y = game_y
        panel_w = PANEL_Q_WIDTH
        panel_h = GAME_H

        # draw game centered
        draw_grid(screen, env, game_x, game_y)
        draw_snake(screen, env, game_x, game_y)
        draw_food(screen, env, game_x, game_y)

        # draw q panel on the right
        draw_q_panel(screen, q_t, action, state, score, fps_text, panel_x, panel_y, panel_w, panel_h)

        pygame.display.flip()

        state = next_state
        if done:
            time.sleep(0.35)
            state = env.reset()
            score = 0

    pygame.quit()
    sys.exit()
