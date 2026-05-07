from aiosmtplib import send
from email.message import EmailMessage
from app.core.config import settings

async def send_verification_email(email_to: str, token: str):
    # SMTP_USER와 SMTP_PASSWORD가 설정되어 있을 때만 메일 발송 시도
    if settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD:
        message = EmailMessage()
        message["From"] = f"{settings.EMAILS_FROM_NAME} <{settings.EMAILS_FROM_EMAIL}>"
        message["To"] = email_to
        message["Subject"] = "Email Verification - Auth Service"
        
        verification_link = f"http://localhost:8000/api/v1/auth/verify-email?token={token}"
        
        content = f"""
        Hello,
        
        Please verify your email by clicking the link below:
        {verification_link}
        """
        message.set_content(content)
        
        try:
            await send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                use_tls=settings.SMTP_TLS,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASSWORD,
            )
            print(f"INFO: Verification email sent to {email_to}")
        except Exception as e:
            print(f"ERROR: Failed to send email: {e}")
    else:
        # 테스트 환경용 로그 출력
        print(f"DEBUG: [TEST MODE] Verification token for {email_to} is {token}")
        print(f"DEBUG: Skip sending email because SMTP is not configured.")
