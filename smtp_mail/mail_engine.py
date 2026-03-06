import smtplib
import imaplib
import email
from email.message import EmailMessage
from email.header import decode_header

class MailEngine:
    def __init__(self, email_address, password, imap_server, smtp_server):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.smtp_server = smtp_server

    def send_email(self, to_address, subject, content):
        """sends a message with SMTP"""
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.email_address
            msg['To'] = to_address
            msg.set_content(content)

            with smtplib.SMTP_SSL(self.smtp_server, 465) as smtp:
                smtp.login(self.email_address, self.password)
                smtp.send_message(msg)
            print(f"Wiadomosc wyslana do {to_address}")
            return True
        except Exception as e:
            print(f"Blad podczas wysylania: {e}")
            return False

    def fetch_emails(self, limit=5):
        """fetches newest emails"""
        emails_data = []
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server, 993)
            mail.login(self.email_address, self.password)
            mail.select("INBOX")
            
            status, messages = mail.search(None, "ALL")
            mail_ids = messages[0].split()
            latest_ids = mail_ids[-limit:]

            for i in reversed(latest_ids):
                status, msg_data = mail.fetch(i, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # decode subject
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8")
                            
                        # decode sender
                        sender, encoding = decode_header(msg.get("From"))[0]
                        if isinstance(sender, bytes):
                            sender = sender.decode(encoding if encoding else "utf-8")

                        body = ""
                        if msg.is_multipart():
                            # Iterate over email parts
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                content_dispo = str(part.get("Content-Disposition"))
                                
                                # ignore attachments
                                if content_type == "text/plain" and "attachment" not in content_dispo:
                                    charset = part.get_content_charset() or "utf-8"
                                    try:
                                        body = part.get_payload(decode=True).decode(charset, errors="replace")
                                    except Exception:
                                        body = "[Could not decode message body]"
                                    break
                        else:
                            charset = msg.get_content_charset() or "utf-8"
                            try:
                                body = msg.get_payload(decode=True).decode(charset, errors="replace")
                            except Exception:
                                body = "[Could not decode message body]"

                        emails_data.append({"sender": sender, "subject": subject, "body": body})
            
            mail.logout()
            print("Wiadomosci pobrane")
            return emails_data
            
        except Exception as e:
            print(f"Pobieranie wiadomosci nie powiodlo sie: {e}")
            return []
