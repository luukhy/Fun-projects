from mail_engine import MailEngine

MOJ_EMAIL = "gpeter080@gmail.com"
MOJE_HASLO = "jibd jctz ddsr ekhw" 
IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"

silnik = MailEngine(MOJ_EMAIL, MOJE_HASLO, IMAP_HOST, SMTP_HOST)


pobrane_maile = silnik.fetch_emails(limit=3)
for mail in pobrane_maile:
    print(f"Od: {mail['sender']} | Temat: {mail['subject']}")