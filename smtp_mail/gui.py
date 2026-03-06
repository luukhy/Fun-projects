import tkinter as tk
from tkinter import messagebox
import threading
from mail_engine import MailEngine
import time


MY_EMAIL = "gpeter080@gmail.com"
MY_PASSWORD = "jibd jctz ddsr ekhw" 
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
        # threading.Thread(target=self.autoresponder, daemon=True).start()
        

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
    
    def autoresponder(self):
        time_s = time.time()
        while True:
            time_now = time.time()
            time_diff = time_now - time_s
            if time_diff > 5:
                curr_emails = self.engine.fetch_emails(limit=10)
                if curr_emails != self.current_emails:
                    senders = [email['sender'] for email in curr_emails] 
                    print(senders)








if __name__ == "__main__":
    root = tk.Tk()
    app = MailClientGUI(root)
    root.mainloop()
