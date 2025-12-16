import pygame
import sys
import os
import torch
import numpy as np
from PIL import Image
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

def record_gameplay(model_path, num_episodes=3, output_dir="recordings", fps=10):
    """Record gameplay and save as images/GIF."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    pygame.init()
    
    CELL_SIZE = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = SnakeEnv(width=20, height=20)
    state_dim = env.reset().shape[0]
    
    agent = DQNAgent(
        input_dim=state_dim,
        output_dim=4,
        device=device,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        memory_capacity=50000,
        use_prioritized_replay=False,
        target_update_steps=1000
    )
    
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Recording {num_episodes} episodes to {output_dir}/")
    
    grid_w = env.width * CELL_SIZE
    grid_h = env.height * CELL_SIZE
    window_w = grid_w + 40
    window_h = grid_h + 140
    
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Recording...")
    clock = pygame.time.Clock()
    
    CAT = {
        "surface0": (20, 20, 26),
        "green": (166, 227, 161),
        "peach": (250, 179, 135),
        "text": (230, 230, 235),
    }
    
    font = pygame.font.SysFont("consolas", 18, bold=True)
    
    for ep in range(1, num_episodes + 1):
        state = env.reset(randomize_start=True)
        frames = []
        step = 0
        
        while True:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.policy(st).cpu().numpy()[0]
            
            action = int(np.argmax(q_values))
            next_state, reward, done = env.step(action)
            state = next_state
            step += 1
            
            # Render
            screen.fill(CAT["surface0"])
            
            # Grid
            for x in range(env.width):
                for y in range(env.height):
                    rect = pygame.Rect(20 + x * CELL_SIZE, 20 + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, (40, 40, 48), rect, 1)
            
            # Snake
            for i, p in enumerate(env.snake):
                color = (200, 255, 190) if i == 0 else CAT["green"]
                rect = pygame.Rect(20 + p.x * CELL_SIZE, 20 + p.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
            
            # Food
            if env.food:
                rect = pygame.Rect(20 + env.food.x * CELL_SIZE, 20 + env.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, CAT["peach"], rect)
            
            # Info
            info_y = grid_h + 50
            screen.blit(font.render(f"Episode: {ep}/{num_episodes}", True, CAT["text"]), (20, info_y))
            screen.blit(font.render(f"Score: {env.score}", True, CAT["text"]), (20, info_y + 25))
            screen.blit(font.render(f"Step: {step}", True, CAT["text"]), (20, info_y + 50))
            
            pygame.display.flip()
            
            # Save frame
            frame_surface = pygame.surfarray.array3d(screen)
            frame_surface = np.transpose(frame_surface, (1, 0, 2))
            frames.append(Image.fromarray(frame_surface))
            
            clock.tick(fps)
            
            if done:
                break
        
        # Save GIF
        gif_path = os.path.join(output_dir, f"episode_{ep}_score_{env.score}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"✓ Saved: {gif_path} (Score: {env.score}, Frames: {len(frames)})")
    
    pygame.quit()
    print(f"\n✓ All recordings saved to '{output_dir}/' directory")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record Snake AI gameplay")
    parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="recordings")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    
    record_gameplay(args.model, args.episodes, args.output, args.fps)
