import os
import smtplib
from email.mime.text import MIMEText
from utils.logger import get_logger

logger = get_logger(__name__)

def send_notification(subject, message):
    sender = "jasonling5555@gmail.com"
    receiver = "jasonling5555@gmail.com"
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, gmail_password)
            server.send_message(msg)
        logger.info("📧 Email notification sent successfully to %s", receiver)
    except Exception as e:
        logger.error("❌ Failed to send email: %s", e)
