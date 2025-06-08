#!/usr/bin/env python3
"""
start_training_with_email.py

Convenient startup script for Chess-ViT training with email notifications.
Handles environment setup and launches training with proper configuration.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_env_variables():
    """Check if required environment variables are set."""
    sender = os.getenv('TRAINING_EMAIL_SENDER')
    password = os.getenv('TRAINING_EMAIL_PASSWORD')
    
    if not sender or not password:
        return False, sender, password
    return True, sender, password

def load_env_file():
    """Try to load environment variables from .env file."""
    env_file = Path('.env')
    if env_file.exists():
        print("Loading environment variables from .env file...")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            return True
        except Exception as e:
            print(f"Error loading .env file: {e}")
            return False
    return False

def main():
    parser = argparse.ArgumentParser(description="Start Chess-ViT training with email notifications")
    parser.add_argument('--config', '-c', default='configs/medium.yaml', 
                       help='Path to training config file')
    parser.add_argument('--email', required=True,
                       help='Email address for daily training updates')
    parser.add_argument('--setup-email', action='store_true',
                       help='Run email setup before starting training')
    parser.add_argument('--test-email', action='store_true',
                       help='Test email configuration and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running training')
    
    args = parser.parse_args()
    
    print("=== Chess-ViT Training with Email Notifications ===\n")
    
    # Step 1: Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        configs_dir = Path('configs')
        if configs_dir.exists():
            for config_file in configs_dir.glob('*.yaml'):
                print(f"  - {config_file}")
        sys.exit(1)
    
    print(f"✅ Using config: {config_path}")
    
    # Step 2: Setup email if requested
    if args.setup_email:
        print("\n--- Running Email Setup ---")
        result = subprocess.run([sys.executable, 'setup_email.py'], capture_output=False)
        if result.returncode != 0:
            print("❌ Email setup failed")
            sys.exit(1)
    
    # Step 3: Check environment variables
    print("\n--- Checking Email Configuration ---")
    
    # Try to load from .env file if needed
    env_loaded, sender, password = check_env_variables()
    if not env_loaded:
        print("Environment variables not found, trying .env file...")
        if load_env_file():
            env_loaded, sender, password = check_env_variables()
    
    if not env_loaded:
        print("❌ Email credentials not found!")
        print("Either:")
        print("1. Run: python setup_email.py")
        print("2. Set environment variables:")
        print("   export TRAINING_EMAIL_SENDER=your_email@gmail.com")
        print("   export TRAINING_EMAIL_PASSWORD=your_app_password")
        print("3. Use --setup-email flag to configure now")
        sys.exit(1)
    
    print(f"✅ Email sender: {sender}")
    print(f"✅ Email recipient: {args.email}")
    
    # Step 4: Test email if requested
    if args.test_email:
        print("\n--- Testing Email Configuration ---")
        result = subprocess.run([
            sys.executable, 'setup_email.py', '--test-only', 
            '--recipient', args.email
        ], capture_output=False)
        if result.returncode == 0:
            print("✅ Email test successful!")
        else:
            print("❌ Email test failed!")
        sys.exit(result.returncode)
    
    # Step 5: Prepare training command
    training_cmd = [
        sys.executable, 'train.py',
        '--config', str(config_path),
        '--email', args.email
    ]
    
    print(f"\n--- Training Configuration ---")
    print(f"Config: {config_path}")
    print(f"Email updates: {args.email} (daily at 10:00 PM)")
    print(f"Command: {' '.join(training_cmd)}")
    
    if args.dry_run:
        print("\n✅ Dry run complete. Use without --dry-run to start training.")
        sys.exit(0)
    
    # Step 6: Start training
    print(f"\n--- Starting Training ---")
    print("Training will run in the background with email notifications.")
    print("Press Ctrl+C to interrupt training.")
    print("=" * 60)
    
    try:
        # Run training
        subprocess.run(training_cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 