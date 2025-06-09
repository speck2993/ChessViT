"""
email_notifier.py

Email notification system for Chess-ViT training updates.
Sends daily training progress reports with loss plots attached.
"""
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Any, Optional
import glob
from pathlib import Path

class TrainingEmailNotifier:
    """Handles email notifications for training updates."""
    
    def __init__(
        self,
        recipient_email: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        send_time_hour: int = 22,  # 10:00 PM
        send_time_minute: int = 0
    ):
        self.recipient_email = recipient_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv('TRAINING_EMAIL_SENDER')
        self.sender_password = sender_password or os.getenv('TRAINING_EMAIL_PASSWORD')
        self.send_time_hour = send_time_hour
        self.send_time_minute = send_time_minute
        self.last_email_date = None
        
        if not self.sender_email or not self.sender_password:
            logging.warning("Email credentials not provided. Email notifications will be disabled.")
            logging.warning("Set TRAINING_EMAIL_SENDER and TRAINING_EMAIL_PASSWORD environment variables or pass them directly.")
        
        logging.info(f"Email notifier initialized. Daily emails will be sent at {send_time_hour:02d}:{send_time_minute:02d}")
    
    def is_credentials_available(self) -> bool:
        """Check if email credentials are available."""
        return bool(self.sender_email and self.sender_password)
    
    def should_send_email_now(self) -> bool:
        """Check if it's time to send the daily email."""
        if not self.is_credentials_available():
            return False
            
        now = datetime.now()
        today_send_time = now.replace(
            hour=self.send_time_hour, 
            minute=self.send_time_minute, 
            second=0, 
            microsecond=0
        )
        
        # Log current time info for debugging
        logging.debug(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.debug(f"Target send time today: {today_send_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.debug(f"Last email date: {self.last_email_date}")
        
        # Check if we've already sent an email today
        if self.last_email_date and self.last_email_date.date() == now.date():
            logging.debug("Email already sent today, skipping")
            return False
        
        # Handle edge case: if target send time was earlier today but we missed it,
        # still send the email (e.g., training started late)
        if now >= today_send_time:
            logging.debug("Current time is past target send time, should send email")
            return True
        
        # If it's past midnight but before the send time, check if we should send yesterday's email
        yesterday = now - timedelta(days=1)
        yesterday_send_time = yesterday.replace(
            hour=self.send_time_hour,
            minute=self.send_time_minute,
            second=0,
            microsecond=0
        )
        
        # If we haven't sent an email since yesterday's send time and it's now past that time
        if (self.last_email_date is None or 
            self.last_email_date < yesterday_send_time) and now >= yesterday_send_time:
            logging.debug("Should send yesterday's missed email")
            return True
        
        logging.debug("Not time to send email yet")
        return False
    
    def create_training_report_email(
        self,
        step: int,
        max_steps: int,
        mean_metrics: Dict[str, float],
        throughput: float,
        output_dir: str,
        training_start_time: Optional[datetime] = None
    ) -> MIMEMultipart:
        """Create email with training progress report and plots."""
        
        msg = MIMEMultipart('related')
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        msg['Subject'] = f"Chess-ViT Training Update - Step {step:,}/{max_steps:,}"
        
        # Calculate progress percentage
        progress_pct = (step / max_steps) * 100 if max_steps > 0 else 0
        
        # Calculate estimated time remaining
        eta_str = "Unknown"
        if training_start_time and step > 0:
            elapsed = datetime.now() - training_start_time
            steps_per_second = step / elapsed.total_seconds()
            remaining_steps = max_steps - step
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            eta_delta = timedelta(seconds=eta_seconds)
            eta_str = str(eta_delta).split('.')[0]  # Remove microseconds
        
        # Create HTML email body
        html_body = f"""
        <html>
        <head></head>
        <body>
            <h2>Chess-ViT Training Progress Report</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Training Status</h3>
            <ul>
                <li><strong>Current Step:</strong> {step:,} / {max_steps:,} ({progress_pct:.1f}%)</li>
                <li><strong>Throughput:</strong> {throughput:.1f} positions/second</li>
                <li><strong>Estimated Time Remaining:</strong> {eta_str}</li>
            </ul>
            
            <h3>Current Loss Metrics</h3>
            <ul>
                <li><strong>Total Loss:</strong> {mean_metrics.get('total', 0):.4f}</li>
                <li><strong>LC0 Comparable Loss:</strong> {mean_metrics.get('compare_lc0', 0):.4f}</li>
                <li><strong>Policy Loss:</strong> {mean_metrics.get('policy', 0):.4f}</li>
                <li><strong>Value Loss:</strong> {mean_metrics.get('value', 0):.4f}</li>
                <li><strong>Moves Left Loss:</strong> {mean_metrics.get('moves_left', 0):.4f}</li>
                <li><strong>Auxiliary Value Loss:</strong> {mean_metrics.get('auxiliary_value', 0):.4f}</li>
                <li><strong>Material Loss:</strong> {mean_metrics.get('material', 0):.4f}</li>
            </ul>
            
            <h3>Loss History Plots</h3>
            <p>See attached plots for detailed loss trends over training.</p>
            
            <p><em>This is an automated email from your Chess-ViT training run.</em></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach loss plots
        plot_files = self._find_plot_files(output_dir)
        for plot_file in plot_files:
            try:
                with open(plot_file, 'rb') as f:
                    img_data = f.read()
                    img = MIMEImage(img_data)
                    img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(plot_file)}"')
                    msg.attach(img)
                    logging.info(f"Attached plot: {os.path.basename(plot_file)}")
            except Exception as e:
                logging.warning(f"Failed to attach plot {plot_file}: {e}")
        
        return msg
    
    def _find_plot_files(self, output_dir: str) -> List[str]:
        """Find all loss plot files in the output directory."""
        plot_patterns = [
            "total_history.png",
            "compare_lc0_history.png",
            "policy_history.png",
            "value_history.png",
            "moves_left_history.png",
            "auxiliary_value_history.png",
            "material_history.png"
        ]
        
        plot_files = []
        for pattern in plot_patterns:
            full_pattern = os.path.join(output_dir, pattern)
            matches = glob.glob(full_pattern)
            plot_files.extend(matches)
        
        # Also look for any other *_history.png files
        general_pattern = os.path.join(output_dir, "*_history.png")
        all_plots = glob.glob(general_pattern)
        for plot in all_plots:
            if plot not in plot_files:
                plot_files.append(plot)
        
        return sorted(plot_files)
    
    def send_email(self, msg: MIMEMultipart) -> bool:
        """Send the email message."""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.last_email_date = datetime.now()
            logging.info(f"Training update email sent successfully to {self.recipient_email}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email: {e}")
            return False
    
    def send_training_update(
        self,
        step: int,
        max_steps: int,
        mean_metrics: Dict[str, float],
        throughput: float,
        output_dir: str,
        training_start_time: Optional[datetime] = None
    ) -> bool:
        """Send training update email if it's time to do so."""
        logging.debug(f"Checking if should send email at step {step}")
        
        if not self.should_send_email_now():
            logging.debug("Email sending conditions not met, skipping")
            return False
        
        logging.info(f"Sending training update email at step {step}")
        
        try:
            msg = self.create_training_report_email(
                step, max_steps, mean_metrics, throughput, output_dir, training_start_time
            )
            success = self.send_email(msg)
            if success:
                logging.info(f"Training update email successfully sent at step {step}")
            else:
                logging.error(f"Failed to send training update email at step {step}")
            return success
        except Exception as e:
            logging.error(f"Failed to create or send training update email: {e}")
            return False
    
    def send_start_notification(
        self,
        config_path: str,
        max_steps: int
    ) -> bool:
        """Sends an email notification when training starts."""
        if not self.is_credentials_available():
            logging.warning("Cannot send start notification: email credentials not available.")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"✅ Chess-ViT Training Run Has Started ({os.path.basename(config_path)})"

            html_body = f"""
            <html>
            <head></head>
            <body>
                <h2>Chess-ViT Training Has Started</h2>
                <p>This is a confirmation that your training run has successfully started.</p>
                <hr>
                
                <h3>Configuration Details</h3>
                <ul>
                    <li><strong>Start Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><strong>Config File:</strong> {config_path}</li>
                    <li><strong>Total Steps:</strong> {max_steps:,}</li>
                </ul>
                
                <p>You will receive your first scheduled update at approximately <strong>{self.send_time_hour:02d}:{self.send_time_minute:02d}</strong>.</p>
                <p><em>Good luck with your training run!</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            logging.info("Sending start-of-training notification...")
            # Reuse the main send_email method which contains all error handling
            return self.send_email(msg)
            
        except Exception as e:
            logging.error(f"Failed to create or send start notification email: {e}")
            return False
    
    def send_training_complete_email(
        self,
        final_step: int,
        final_metrics: Dict[str, float],
        output_dir: str,
        training_start_time: Optional[datetime] = None
    ) -> bool:
        """Send final training completion email."""
        if not self.is_credentials_available():
            return False
        
        try:
            msg = MIMEMultipart('related')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Chess-ViT Training COMPLETED - Final Step {final_step:,}"
            
            total_time = "Unknown"
            if training_start_time:
                elapsed = datetime.now() - training_start_time
                total_time = str(elapsed).split('.')[0]  # Remove microseconds
            
            html_body = f"""
            <html>
            <head></head>
            <body>
                <h2>🎉 Chess-ViT Training Completed!</h2>
                <p><strong>Completion Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h3>Final Results</h3>
                <ul>
                    <li><strong>Final Step:</strong> {final_step:,}</li>
                    <li><strong>Total Training Time:</strong> {total_time}</li>
                </ul>
                
                <h3>Final Loss Metrics</h3>
                <ul>
                    <li><strong>Total Loss:</strong> {final_metrics.get('total', 0):.4f}</li>
                    <li><strong>LC0 Comparable Loss:</strong> {final_metrics.get('compare_lc0', 0):.4f}</li>
                    <li><strong>Policy Loss:</strong> {final_metrics.get('policy', 0):.4f}</li>
                    <li><strong>Value Loss:</strong> {final_metrics.get('value', 0):.4f}</li>
                    <li><strong>Moves Left Loss:</strong> {final_metrics.get('moves_left', 0):.4f}</li>
                    <li><strong>Auxiliary Value Loss:</strong> {final_metrics.get('auxiliary_value', 0):.4f}</li>
                    <li><strong>Material Loss:</strong> {final_metrics.get('material', 0):.4f}</li>
                </ul>
                
                <h3>Final Loss Plots</h3>
                <p>See attached final loss trend plots.</p>
                
                <p><em>Congratulations on completing your 5-week training run!</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach final plots
            plot_files = self._find_plot_files(output_dir)
            for plot_file in plot_files:
                try:
                    with open(plot_file, 'rb') as f:
                        img_data = f.read()
                        img = MIMEImage(img_data)
                        img.add_header('Content-Disposition', f'attachment; filename="FINAL_{os.path.basename(plot_file)}"')
                        msg.attach(img)
                except Exception as e:
                    logging.warning(f"Failed to attach final plot {plot_file}: {e}")
            
            return self.send_email(msg)
            
        except Exception as e:
            logging.error(f"Failed to send training completion email: {e}")
            return False
    
    def debug_email_status(self) -> Dict[str, Any]:
        """Return detailed information about email timing status for debugging."""
        now = datetime.now()
        today_send_time = now.replace(
            hour=self.send_time_hour, 
            minute=self.send_time_minute, 
            second=0, 
            microsecond=0
        )
        
        yesterday = now - timedelta(days=1)
        yesterday_send_time = yesterday.replace(
            hour=self.send_time_hour,
            minute=self.send_time_minute,
            second=0,
            microsecond=0
        )
        
        tomorrow = now + timedelta(days=1)
        tomorrow_send_time = tomorrow.replace(
            hour=self.send_time_hour,
            minute=self.send_time_minute,
            second=0,
            microsecond=0
        )
        
        should_send = self.should_send_email_now()

        status = {
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'timezone_note': 'Using local system time (should be UTC-7 in your case)',
            'target_send_time_today': today_send_time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_send_time_yesterday': yesterday_send_time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_send_time_tomorrow': tomorrow_send_time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_email_sent': self.last_email_date.strftime('%Y-%m-%d %H:%M:%S') if self.last_email_date else 'Never',
            'credentials_available': self.is_credentials_available(),
            'should_send_now': should_send,
            'next_email_update': None,
            'days_since_last_email': None
        }
        
        # Calculate time until/since next send opportunity
        if now >= today_send_time:
            if should_send:
                time_since_due = now - today_send_time
                status['next_email_update'] = f"Overdue by {str(time_since_due).split('.')[0]}"
            else:  # Already sent today
                time_until = tomorrow_send_time - now
                status['next_email_update'] = f"in {str(time_until).split('.')[0]}"
        else:
            # Next opportunity is today
            time_until = today_send_time - now
            status['next_email_update'] = f"in {str(time_until).split('.')[0]}"
        
        # Calculate days since last email
        if self.last_email_date:
            days_since = (now.date() - self.last_email_date.date()).days
            status['days_since_last_email'] = days_since
        
        return status
    
    def print_debug_status(self):
        """Print detailed email timing status for debugging."""
        status = self.debug_email_status()
        print("\n=== EMAIL NOTIFICATION DEBUG STATUS ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        print("=" * 40)


def setup_email_logging():
    """Setup logging for email notifications."""
    # Create a separate logger for email notifications
    email_logger = logging.getLogger('email_notifier')
    email_logger.setLevel(logging.DEBUG)  # Changed to DEBUG for more detailed logging
    
    # Create console handler if it doesn't exist
    if not email_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Changed to DEBUG
        formatter = logging.Formatter('[EMAIL] %(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        email_logger.addHandler(console_handler)
    
    # Also set up the root logger to show email debug messages
    root_logger = logging.getLogger()
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.INFO)  # Ensure at least INFO level for root logger
    
    return email_logger


if __name__ == "__main__":
    # Quick test of email functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="Test email notifications")
    parser.add_argument('--recipient', required=True, help='Recipient email address')
    parser.add_argument('--sender', help='Sender email address (or set TRAINING_EMAIL_SENDER env var)')
    parser.add_argument('--password', help='Sender email password (or set TRAINING_EMAIL_PASSWORD env var)')
    args = parser.parse_args()
    
    setup_email_logging()
    
    notifier = TrainingEmailNotifier(
        recipient_email=args.recipient,
        sender_email=args.sender,
        sender_password=args.password
    )
    
    if notifier.is_credentials_available():
        # Send test email
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
        print(f"Test email sent: {'Success' if success else 'Failed'}")
    else:
        print("Email credentials not available. Please set environment variables or pass them as arguments.") 