import sys
sys.path.append("/opt/airflow")

from utils.send_notifications import send_notification

subject = "Test Fraud Alert 🚨"
message = "This is a test email from the Airflow container."

send_notification(subject, message)
