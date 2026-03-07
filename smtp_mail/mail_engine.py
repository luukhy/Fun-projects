import smtplib
import imaplib
import email
import email.utils
from email.message import EmailMessage
from email.header import decode_header
import json

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
            return emails_data
            
        except Exception as e:
            print(f"Pobieranie wiadomosci nie powiodlo sie: {e}")
            return []

    def run_autoresponder(self, reply_subject, reply_body, limit=20):
        with open('autoresponder.json', 'r') as auto_file:
            auto_data = json.load(auto_file)

        target_addrs = auto_data['target_emails']

        replied_count = 0
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server, 993)
            mail.login(self.email_address, self.password)
            mail.select("INBOX")

            _, message = mail.search(None, "UNSEEN")
            mail_ids = message[0].split()

            if not mail_ids:
                mail.logout()
                return 0

            recent_ids = mail_ids[-limit:]
            for i in recent_ids:
                _, msg_data = mail.fetch(i, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        raw_sender, encoding = decode_header(msg.get("From"))[0]

                        if isinstance(raw_sender, bytes):
                            raw_sender = raw_sender.decode(encoding if encoding else "utf-8")
                        
                        _, sender_email = email.utils.parseaddr(raw_sender)

                        print(f"Checking email: {sender_email}") # Debug print

                        if sender_email not in target_addrs:
                            continue

                        if sender_email:
                            print(f"Sent an email to {sender_email}")
                            self.send_email(sender_email, reply_subject, reply_body)
                            replied_count += 1
            mail.logout
            return replied_count

        except Exception as e:
            print(f"Autoresponder error {e}")
            return 0
