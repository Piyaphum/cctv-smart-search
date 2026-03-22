"""
Email Service for sending alerts
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import SENDER_EMAIL, SENDER_PASSWORD, WEB_APP_URL


def send_email_report(summary_dict, recipient_emails, sender_email=SENDER_EMAIL, sender_password=SENDER_PASSWORD):
    """
    Send email report with detection summary
    
    Args:
        summary_dict: {video_name: {target_name: [(match_data), ...]}}
        recipient_emails: list of email addresses
        sender_email: sender email
        sender_password: sender password
    
    Returns:
        (success, message)
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Detection Alert: Found Matches in {len(summary_dict)} Videos'
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipient_emails)
        
        report_html = ""
        total_matches = 0
        
        for video_name, targets in summary_dict.items():
            report_html += f"<h3>{video_name}</h3><ul>"
            for target_name, logs in targets.items():
                count = len(logs)
                total_matches += count
                color = logs[0]["color"] if logs else "Unknown"
                gender = logs[0]["gender"] if logs and "gender" in logs[0] else "Unknown"
                report_html += f"<li>Found <b>{target_name}</b> {count}x (Clothing: {color}, Gender: {gender})</li>"
            report_html += "</ul><hr>"

        html_body = f"""
        <html>
            <body style="font-family:Arial, sans-serif; background-color:#0a0e27; color:#e8eef2;">
                <div style="background-color:#151b28; padding:20px; border-radius:8px; border-left:4px solid #1dd1a1;">
                    <h2 style="color:#1dd1a1; margin:0;">Detection Alert</h2>
                    <p style="margin:10px 0 0;">Total matches found: <b>{total_matches}</b></p>
                </div>
                <div style="margin-top:20px;">
                    {report_html}
                </div>
                <a href="{WEB_APP_URL}" style="background-color:#1dd1a1; color:#000; padding:12px 20px; text-decoration:none; border-radius:6px; font-weight:bold;">
                    View Results
                </a>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)


def send_password_reset_email(recipient_email, username, new_password, sender_email=SENDER_EMAIL, sender_password=SENDER_PASSWORD):
    """
    Send password reset email
    
    Args:
        recipient_email: user's registered email
        username: user's username
        new_password: newly generated password
        sender_email: sender email
        sender_password: sender password
        
    Returns:
        (success, message)
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Password Reset Completed'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        html_body = f"""
        <html>
            <body style="font-family:Arial, sans-serif; background-color:#0a0e27; color:#e8eef2;">
                <div style="background-color:#151b28; padding:20px; border-radius:8px; border-left:4px solid #1dd1a1;">
                    <h2 style="color:#1dd1a1; margin:0;">Password Reset Successfully</h2>
                    <p style="margin:10px 0 0;">Hello <b>{username}</b>,</p>
                    <p>Your password for the Person Detection System has been successfully reset.</p>
                    <p>Your new temporary password is: <b style="background-color:#2c3e50; padding:4px 8px; border-radius:4px;">{new_password}</b></p>
                    <p style="margin-top:20px; font-size:12px; color:#a0aec0;">If you did not request this change, please contact your system administrator.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)


def send_verification_code_email(recipient_email, username, verification_code, sender_email=SENDER_EMAIL, sender_password=SENDER_PASSWORD):
    """
    Send a 6-digit OTP verification code for password reset
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Password Reset Verification Code'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        html_body = f"""
        <html>
            <body style="font-family:Arial, sans-serif; background-color:#0a0e27; color:#e8eef2;">
                <div style="background-color:#151b28; padding:20px; border-radius:8px; border-left:4px solid #1dd1a1;">
                    <h2 style="color:#1dd1a1; margin:0;">Verification Code</h2>
                    <p style="margin:10px 0 0;">Hello <b>{username}</b>,</p>
                    <p>We received a request to reset your password. Please use the verification code below to complete the process:</p>
                    <div style="font-size: 24px; font-weight: bold; margin: 20px 0; background-color: #2c3e50; padding: 10px; text-align: center; letter-spacing: 5px; border-radius: 5px;">
                        {verification_code}
                    </div>
                    <p style="margin-top:20px; font-size:12px; color:#a0aec0;">If you did not request this code, please ignore this email. Your password will remain unchanged.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)
