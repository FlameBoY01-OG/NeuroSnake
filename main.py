#!/usr/bin/env python
# main.py - CLI interface for NeuroSnake
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="NeuroSnake - Deep Q-Learning Snake AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train
  
  # Train with custom episodes
  python main.py train --episodes 2000
  
  # Watch trained agent (dual-window: game + neural network)
  python main.py play
  
  # Watch specific checkpoint with faster speed
  python main.py play --model model_checkpoints/policy_ep800.pth --fps 20
  
  # Evaluate model performance
  python main.py eval
  
  # Compare all checkpoints
  python main.py eval --compare
  
  # Test environment
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--episodes", type=int, default=1000,
                              help="Number of training episodes (default: 1000)")
    train_parser.add_argument("--lr", type=float, default=1e-3,
                              help="Learning rate (default: 0.001)")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="Batch size (default: 64)")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Watch agent with dual-window visualization (game + neural net)")
    play_parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth",
                             help="Path to model checkpoint")
    play_parser.add_argument("--fps", type=int, default=14,
                             help="Game speed (frames per second)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth",
                             help="Path to model checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=100,
                             help="Number of evaluation episodes")
    eval_parser.add_argument("--compare", action="store_true",
                             help="Compare all available checkpoints")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test environment and setup")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate training plots")
    plot_parser.add_argument("--log", type=str, default="training_log.txt", help="Training log file")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record gameplay as GIF")
    record_parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth")
    record_parser.add_argument("--episodes", type=int, default=3)
    record_parser.add_argument("--output", type=str, default="recordings")
    record_parser.add_argument("--fps", type=int, default=10)
    
    args = parser.parse_args()
    
    if not args.command:
        # Default to play mode if model exists, otherwise show help
        if os.path.exists("model_checkpoints/policy_final.pth"):
            print("No command specified - launching visualization (use --help to see all commands)\n")
            args.command = "play"
            args.model = "model_checkpoints/policy_final.pth"
            args.fps = 14
        else:
            parser.print_help()
            sys.exit(1)
    
    # Execute commands
    if args.command == "train":
        from training.train import train
        print(f"Starting training for {args.episodes} episodes...")
        train(num_episodes=args.episodes)
        
    elif args.command == "play":
        print(f"Launching complete visualization with model: {args.model}")
        os.environ['MODEL_PATH'] = args.model
        os.environ['FPS'] = str(getattr(args, 'fps', 14))
        import subprocess
        subprocess.run([sys.executable, "-m", "visualization.complete_view"])
        
    elif args.command == "eval":
        from training.evaluate import evaluate_model, print_statistics, compare_checkpoints
        if args.compare:
            print("Comparing all checkpoints...")
            compare_checkpoints(num_episodes=args.episodes)
        else:
            print(f"Evaluating model: {args.model}")
            stats = evaluate_model(args.model, num_episodes=args.episodes)
            print_statistics(stats)
            
    elif args.command == "test":
        print("Testing NeuroSnake environment...\n")
        from env.snake_env import SnakeEnv
        import torch
        
        # Test environment
        env = SnakeEnv(width=20, height=20)
        state = env.reset()
        print(f"✓ Environment initialized")
        print(f"  Grid size: {env.width}x{env.height}")
        print(f"  State shape: {state.shape}")
        print(f"  State features: {len(state)}")
        
        # Test a few steps
        print(f"\n✓ Running 5 test steps...")
        for i in range(5):
            action = i % 4
            next_state, reward, done = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.1f}, done={done}")
            if done:
                state = env.reset()
            else:
                state = next_state
        
        # Test PyTorch
        print(f"\n✓ PyTorch info:")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        
        print("\n✓ All tests passed! Environment is ready.\n")
    
    elif args.command == "plot":
        from visualization.plot_results import plot_training_results
        if os.path.exists(args.log):
            print(f"Generating plots from {args.log}...")
            plot_training_results(args.log)
            print("✓ Plots saved to plots/ directory")
        else:
            print(f"❌ Error: {args.log} not found!")
            print("   Run training first with: python main.py train")
            print("   Training will create the log file automatically")
    
    elif args.command == "record":
        from visualization.record_gameplay import record_gameplay
        print(f"Recording {args.episodes} episodes to {args.output}/")
        record_gameplay(args.model, args.episodes, args.output, args.fps)

if __name__ == "__main__":
    main()

