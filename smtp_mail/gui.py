import tkinter as tk
from tkinter import messagebox
import threading
from mail_engine import MailEngine
import time


MY_EMAIL = "gpeter080@gmail.com"
MY_PASSWORD = "plun jmkl jckb wucj" 
IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"

class MailClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("My Email Client")
        self.root.geometry("650x700") 
        
        self.engine = MailEngine(MY_EMAIL, MY_PASSWORD, IMAP_HOST, SMTP_HOST)
        self.current_emails = []

        self.build_interface()
        self.refresh_in_background()

        self.autoresponder_active = False

    def build_interface(self):
        self.receive_frame = tk.LabelFrame(self.root, text="Inbox", padx=10, pady=10)
        self.receive_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        list_frame = tk.Frame(self.receive_frame)
        list_frame.pack(fill="both", expand=True, pady=5)
        
        self.list_scrollbar = tk.Scrollbar(list_frame)
        self.list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.email_listbox = tk.Listbox(list_frame, height=6, width=78, yscrollcommand=self.list_scrollbar.set)
        self.email_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        self.list_scrollbar.config(command=self.email_listbox.yview)
        
        self.email_listbox.bind('<<ListboxSelect>>', self.on_email_select)
        
        self.refresh_btn = tk.Button(self.receive_frame, text="Odswiez", command=self.refresh_in_background)
        self.refresh_btn.pack(pady=5)

        # message content
        tk.Label(self.receive_frame, text="Tresc wiadomosci:").pack(anchor="w")
        
        msg_frame = tk.Frame(self.receive_frame)
        msg_frame.pack(fill="both", expand=True, pady=5)
        
        self.msg_scrollbar = tk.Scrollbar(msg_frame)
        self.msg_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.message_view = tk.Text(msg_frame, height=8, width=78, state=tk.DISABLED, yscrollcommand=self.msg_scrollbar.set)
        self.message_view.pack(side=tk.LEFT, fill="both", expand=True)
        self.msg_scrollbar.config(command=self.message_view.yview)

        # send section
        self.send_frame = tk.LabelFrame(self.root, text="Wyslij nowa wiadomosc", padx=10, pady=10)
        self.send_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(self.send_frame, text="Odbiorca:").grid(row=0, column=0, sticky="e", pady=2)
        self.to_entry = tk.Entry(self.send_frame, width=65)
        self.to_entry.grid(row=0, column=1, pady=2, sticky="w")

        tk.Label(self.send_frame, text="Temat:").grid(row=1, column=0, sticky="e", pady=2)
        self.subject_entry = tk.Entry(self.send_frame, width=65)
        self.subject_entry.grid(row=1, column=1, pady=2, sticky="w")

        tk.Label(self.send_frame, text="Tresc:").grid(row=2, column=0, sticky="ne", pady=2)
        self.body_text = tk.Text(self.send_frame, width=65, height=6) # Slightly shorter to fit well
        self.body_text.grid(row=2, column=1, pady=2)

        self.send_btn = tk.Button(self.send_frame, text="Wyslij wiadomosc", command=self.send_in_background)
        self.send_btn.grid(row=3, column=1, pady=10, sticky="e")

        # autoresponder
        self.auto_frame = tk.LabelFrame(self.root, text="Autoresponder", padx=10, pady=10)
        self.auto_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(self.auto_frame, text="Temat automatycznej odpowiedzi:").grid(row=0, column=0, sticky="e")
        self.auto_subject = tk.Entry(self.auto_frame, width=40)
        self.auto_subject.insert(0, "Automatyczna odpowiedz: Aktualnie mnie nie ma")
        self.auto_subject.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        tk.Label(self.auto_frame, text="Tresc automatycznej odpowiedzi:").grid(row=1, column=0, sticky="ne")
        self.auto_body = tk.Text(self.auto_frame, width=40, height=3)
        self.auto_body.insert("1.0", "Hej, aktualnie mnie nie ma. Odpowiem po powrocie")
        self.auto_body.grid(row=1, column=1, padx=5, pady=2)

        self.auto_btn = tk.Button(self.auto_frame, text="Uruchom Autoresponder", bg="lightgreen", command=self.toggle_autoresponder)
        self.auto_btn.grid(row=0, column=2, rowspan=2, padx=15)

    def update_email_list(self, emails):
        self.email_listbox.delete(0, tk.END)
        self.current_emails = emails
        
        if not emails:
            self.email_listbox.insert(tk.END, "Brak wiadomosci")
            return
            
        for email in emails:
            display_text = f"Nadawca: {email['sender']} | Temat: {email['subject']}"
            self.email_listbox.insert(tk.END, display_text)

    def fetch_task(self):
        emails = self.engine.fetch_emails(limit=20) 
        self.root.after(0, self.update_email_list, emails)

    def refresh_in_background(self):
        self.email_listbox.delete(0, tk.END)
        self.email_listbox.insert(tk.END, "Pobieranie wiadomosci...")
        threading.Thread(target=self.fetch_task, daemon=True).start()

    def on_email_select(self, event):
        selection = self.email_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        if index < len(self.current_emails):
            selected_email = self.current_emails[index]
            
            self.message_view.config(state=tk.NORMAL)
            self.message_view.delete("1.0", tk.END)
            self.message_view.insert(tk.END, selected_email.get("body", "No content available."))
            self.message_view.config(state=tk.DISABLED)

    def send_in_background(self):
        to_address = self.to_entry.get()
        subject = self.subject_entry.get()
        body = self.body_text.get("1.0", tk.END).strip()

        if not to_address or not subject:
            messagebox.showwarning("Error", "Uzupelnij wszystkie pola")
            return

        threading.Thread(target=self.send_task, args=(to_address, subject, body), daemon=True).start()

    def send_task(self, to_address, subject, body):
        success = self.engine.send_email(to_address, subject, body)
        if success:
            self.root.after(0, lambda: messagebox.showinfo("Sukces", "Wiadomosc wyslana z powodzeniem"))
            self.root.after(0, self.clear_form)
        else:
            self.root.after(0, lambda: messagebox.showerror("Blad", "Nie udalo sie wyslac wiadomosci"))

    def clear_form(self):
        self.to_entry.delete(0, tk.END)
        self.subject_entry.delete(0, tk.END)
        self.body_text.delete("1.0", tk.END)
    
    def toggle_autoresponder(self):
        if not self.autoresponder_active:
            self.autoresponder_active = True
            self.auto_btn.config(text="Stop Autoresponder", bg="salmon")
            self.auto_subject.config(state=tk.DISABLED)
            self.auto_body.config(state=tk.DISABLED)
            
            threading.Thread(target=self._autoresponder_loop, daemon=True).start()
        else:
            self.autoresponder_active = False 
            self.auto_btn.config(text="Start Autoresponder", bg="lightgreen")
            self.auto_subject.config(state=tk.NORMAL)
            self.auto_body.config(state=tk.NORMAL)

    def _autoresponder_loop(self):
        while self.autoresponder_active:
            subject = self.auto_subject.get()
            body = self.auto_body.get("1.0", tk.END).strip()

            replied_to = self.engine.run_autoresponder(subject, body)
            if replied_to > 0:
                print(f"Autoresponder wyslal {replied_to} wiadomosci")
                self.root.after(0, self.refresh_in_background)
            
            time.sleep(10)

    
if __name__ == "__main__":
    root = tk.Tk()
    app = MailClientGUI(root)
    root.mainloop()
