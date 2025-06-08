# Email Notifications for Chess-ViT Training

This guide will help you set up automated daily email updates for your 5-week Chess-ViT training run. You'll receive progress reports every day at 10:00 PM with loss plots attached.

## Quick Start

1. **Setup email credentials**:
   ```bash
   python setup_email.py
   ```

2. **Start training with email notifications**:
   ```bash
   # Load email credentials
   export $(grep -v '^#' .env | xargs)
   
   # Start training
   python train.py --config configs/medium.yaml --email your_email@example.com
   ```

## Email Configuration Options

### Option 1: Interactive Setup (Recommended)
```bash
python setup_email.py
```
This will:
- Guide you through entering your email credentials
- Test the configuration by sending a test email
- Save credentials to a `.env` file for easy reuse

### Option 2: Manual Environment Variables
```bash
export TRAINING_EMAIL_SENDER="your_gmail@gmail.com"
export TRAINING_EMAIL_PASSWORD="your_app_password"
```

### Option 3: Command Line Arguments
```bash
python setup_email.py --sender your_gmail@gmail.com --recipient your_email@example.com --password your_app_password
```

## Gmail Setup (Most Common)

For Gmail users, you'll need to create an **App Password**:

1. Go to [Google Account settings](https://myaccount.google.com/)
2. Navigate to **Security** → **2-Step Verification** → **App passwords**
3. Select **Mail** and generate a password
4. Use the 16-character generated password (not your regular Gmail password)

**Important**: Regular Gmail passwords won't work - you must use an App Password.

## What You'll Receive

### Daily Updates (10:00 PM)
- **Subject**: "Chess-ViT Training Update - Step X/Y"
- **Content**:
  - Current training progress (step count, percentage complete)
  - Throughput (positions/second)
  - Estimated time remaining
  - Current loss metrics (total, policy, value, etc.)
  - All loss history plots attached as PNG files

### Training Completion
- **Subject**: "Chess-ViT Training COMPLETED - Final Step X"
- **Content**:
  - Final training results
  - Total training time
  - Final loss metrics
  - Final loss plots with "FINAL_" prefix

### Interruption Handling
If training is interrupted (Ctrl+C), you'll receive a completion email with the current state.

## Email Schedule

- **Time**: 10:00 PM daily (based on your machine's local time)
- **Frequency**: Once per day maximum
- **Trigger**: Checked during regular logging intervals (every 25,000 steps by default)

## Testing Your Setup

### Test Current Configuration
```bash
python setup_email.py --test-only --recipient your_email@example.com
```

### Test with Specific Credentials
```bash
python setup_email.py --test-only \
  --sender your_gmail@gmail.com \
  --recipient your_email@example.com \
  --password your_app_password
```

## Troubleshooting

### Common Issues

**1. "Email credentials not available"**
- Make sure environment variables are set:
  ```bash
  echo $TRAINING_EMAIL_SENDER
  echo $TRAINING_EMAIL_PASSWORD
  ```
- If empty, load from `.env` file:
  ```bash
  export $(grep -v '^#' .env | xargs)
  ```

**2. "Authentication failed"**
- For Gmail: Ensure you're using an App Password, not your regular password
- For other providers: Check username/password are correct
- Verify 2FA is enabled (required for Gmail App Passwords)

**3. "SMTP connection failed"**
- Check your internet connection
- For corporate networks: May need to configure proxy settings
- For other email providers: May need different SMTP settings

**4. "No plots attached"**
- Plots are generated during training - first email may have few/no plots
- Check that `matplotlib: true` is set in your config file
- Verify plots are being saved to the output directory

### Other Email Providers

For non-Gmail providers, you may need to modify SMTP settings in `email_notifier.py`:

```python
# Example for Outlook/Hotmail
smtp_server="smtp-mail.outlook.com"
smtp_port=587

# Example for Yahoo
smtp_server="smtp.mail.yahoo.com" 
smtp_port=587
```

## Security Considerations

- **App Passwords**: Safer than using your main password
- **Environment Variables**: Keep credentials out of code
- **`.env` file**: Add to `.gitignore` to avoid committing credentials
- **Network**: Uses TLS encryption for email transmission

## File Structure

After setup, you'll have:
```
ChessViT/
├── email_notifier.py      # Email notification system
├── setup_email.py         # Setup and testing script
├── train.py              # Modified training script
├── .env                  # Email credentials (if saved)
└── EMAIL_SETUP.md        # This guide
```

## Advanced Configuration

### Custom Email Time
Modify `send_time_hour` in `train.py`:
```python
email_notifier = TrainingEmailNotifier(
    recipient_email=email_address,
    send_time_hour=14,  # 2:00 PM instead of 10:00 PM
    send_time_minute=30  # 14:30 (2:30 PM)
)
```

### Multiple Recipients
Currently supports one recipient. For multiple recipients, modify the `recipient_email` parameter to use a mailing list or modify the code to support multiple addresses.

### Custom SMTP Settings
Modify the `TrainingEmailNotifier` constructor in `train.py`:
```python
email_notifier = TrainingEmailNotifier(
    recipient_email=email_address,
    smtp_server="your.smtp.server.com",
    smtp_port=587,
    send_time_hour=22,
    send_time_minute=0
)
```

## Example Training Command

For your 5-week training run:
```bash
# Setup (once)
python setup_email.py

# Load credentials
export $(grep -v '^#' .env | xargs)

# Start training with email notifications
nohup python train.py \
  --config configs/medium.yaml \
  --email your_email@example.com \
  > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

This setup ensures you'll receive daily updates throughout your 5-week training period, allowing you to monitor progress without being physically present at your machine. 