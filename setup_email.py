#!/usr/bin/env python3
"""
setup_email.py

Helper script to setup and test email notifications for Chess-ViT training.
"""
import os
import getpass
import argparse
from email_notifier import TrainingEmailNotifier, setup_email_logging
from datetime import datetime, timedelta

def get_email_credentials():
    """Interactively get email credentials from user."""
    print("=== Email Setup for Chess-ViT Training Notifications ===\n")
    
    sender = input("Enter your sender email address (e.g., your_gmail@gmail.com): ").strip()
    recipient = input("Enter the recipient email address (where you want updates sent): ").strip()
    
    print(f"\nFor Gmail, you'll need an 'App Password' rather than your regular password.")
    print("To create an App Password:")
    print("1. Go to your Google Account settings")
    print("2. Security > 2-Step Verification > App passwords")
    print("3. Select 'Mail' and generate a password")
    print("4. Use that 16-character password below\n")
    
    password = getpass.getpass("Enter your email password (or App Password for Gmail): ").strip()
    
    return sender, recipient, password

def test_email_configuration(sender, recipient, password):
    """Test the email configuration by sending a test email."""
    print("\n=== Testing Email Configuration ===")
    
    setup_email_logging()
    
    notifier = TrainingEmailNotifier(
        recipient_email=recipient,
        sender_email=sender,
        sender_password=password
    )
    
    if not notifier.is_credentials_available():
        print("❌ Email credentials not properly configured.")
        return False
    
    print(f"Sending test email to {recipient}...")
    
    # Create test training report
    test_metrics = {
        'total': 1.234,
        'compare_lc0': 2.345,
        'policy': 0.123,
        'value': 0.456,
        'moves_left': 0.789,
        'auxiliary_value': 0.321,
        'material': 0.654
    }
    
    msg = notifier.create_training_report_email(
        step=100000,
        max_steps=20000000,
        mean_metrics=test_metrics,
        throughput=1500.0,
        output_dir='./test_output',
        training_start_time=datetime.now() - timedelta(days=2)
    )
    
    success = notifier.send_email(msg)
    
    if success:
        print("✅ Test email sent successfully!")
        print(f"Check your inbox at {recipient}")
        return True
    else:
        print("❌ Failed to send test email. Check your credentials and network connection.")
        return False

def save_credentials_to_env_file(sender, password):
    """Save credentials to a .env file for easy loading."""
    env_content = f"""# Email credentials for Chess-ViT training notifications
TRAINING_EMAIL_SENDER={sender}
TRAINING_EMAIL_PASSWORD={password}
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Credentials saved to .env file")
    print("You can load these environment variables with:")
    print("  export $(grep -v '^#' .env | xargs)")
    print("  # or")
    print("  source .env  # if using bash")

def show_usage_instructions(recipient):
    """Show how to use the email notifications with training."""
    print(f"\n=== Usage Instructions ===")
    print("To start training with email notifications:")
    print(f"  python train.py --config configs/medium.yaml --email {recipient}")
    print()
    print("Make sure to set environment variables first:")
    print("  export TRAINING_EMAIL_SENDER=your_sender@gmail.com")
    print("  export TRAINING_EMAIL_PASSWORD=your_app_password")
    print("  # or load from .env file:")
    print("  export $(grep -v '^#' .env | xargs)")
    print()
    print("Email updates will be sent daily at 10:00 PM with:")
    print("  • Training progress and metrics")
    print("  • Loss history plots attached")
    print("  • Estimated time remaining")
    print("  • Final completion email when training finishes")

def main():
    parser = argparse.ArgumentParser(description="Setup email notifications for Chess-ViT training")
    parser.add_argument('--test-only', action='store_true', 
                       help='Test existing configuration without setting up new credentials')
    parser.add_argument('--sender', help='Sender email address')
    parser.add_argument('--recipient', help='Recipient email address') 
    parser.add_argument('--password', help='Email password (use App Password for Gmail)')
    args = parser.parse_args()
    
    if args.test_only:
        # Test with environment variables or provided args
        sender = args.sender or os.getenv('TRAINING_EMAIL_SENDER')
        recipient = args.recipient or input("Enter recipient email: ").strip()
        password = args.password or os.getenv('TRAINING_EMAIL_PASSWORD')
        
        if not sender or not password:
            print("❌ Sender email and password must be provided via arguments or environment variables")
            print("Set TRAINING_EMAIL_SENDER and TRAINING_EMAIL_PASSWORD environment variables")
            return
        
        success = test_email_configuration(sender, recipient, password)
        if success:
            show_usage_instructions(recipient)
    else:
        # Interactive setup
        if args.sender and args.recipient and args.password:
            sender, recipient, password = args.sender, args.recipient, args.password
        else:
            sender, recipient, password = get_email_credentials()
        
        if not all([sender, recipient, password]):
            print("❌ All email fields are required")
            return
        
        # Test configuration
        success = test_email_configuration(sender, recipient, password)
        
        if success:
            # Save credentials
            save_env = input("\nSave credentials to .env file? (y/n): ").strip().lower()
            if save_env in ['y', 'yes']:
                save_credentials_to_env_file(sender, password)
            
            show_usage_instructions(recipient)
        else:
            print("\n❌ Email configuration failed. Please check your credentials and try again.")

if __name__ == "__main__":
    main() 