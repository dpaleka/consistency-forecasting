import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv


dotenv_path = os.path.dirname(os.path.abspath(os.getcwd()))
dotenv_path = os.path.join(dotenv_path, ".env")
# print(dotenv_path)
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


def send_email(subject, content, recipient_email):
    # Email configuration
    sender_email = os.environ.get("EMAIL")
    sender_password = os.environ.get("EMAIL_PWD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Create the email message
    msg = MIMEText(content)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Connect to the SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print(f"Email sent to {recipient_email}")
