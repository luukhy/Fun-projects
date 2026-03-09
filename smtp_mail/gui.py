import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from mail_engine import MailEngine

MY_EMAIL = "gpeter080@gmail.com"
MY_PASSWORD = "fqme snkk ckhf unlt"
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
        self.displayed_emails = []

    def build_interface(self):
        # inbox
        self.receive_frame = tk.LabelFrame(self.root, text="Inbox", padx=10, pady=10)
        self.receive_frame.pack(fill="both", expand=True, padx=10, pady=10)

        filter_frame = tk.Frame(self.receive_frame)
        filter_frame.pack(fill="x", pady=(0, 5))

        tk.Label(filter_frame, text="Filtruj (slowo kluczowe):").pack(side=tk.LEFT)
        self.filter_entry = tk.Entry(filter_frame, width=20)
        self.filter_entry.pack(side=tk.LEFT, padx=5)

        self.filter_btn = tk.Button(
            filter_frame, text="Szukaj (NLP)", command=self.apply_nlp_filter
        )
        self.filter_btn.pack(side=tk.LEFT)

        self.reset_filter_btn = tk.Button(
            filter_frame, text="Wyczysc filtr", command=self.reset_filter
        )
        self.reset_filter_btn.pack(side=tk.LEFT, padx=5)

        list_frame = tk.Frame(self.receive_frame)
        list_frame.pack(fill="both", expand=True, pady=5)

        self.list_scrollbar = tk.Scrollbar(list_frame)
        self.list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(
            list_frame, columns=("sender", "subject"), show="headings", height=6
        )
        self.tree.heading("sender", text="Nadawca")
        self.tree.heading("subject", text="Temat")
        self.tree.column("sender", width=200)
        self.tree.column("subject", width=400)

        self.tree.tag_configure("unseen", font=("TkDefaultFont", 9, "bold"))

        self.tree.pack(side=tk.LEFT, fill="both", expand=True)
        self.tree.config(yscrollcommand=self.list_scrollbar.set)
        self.list_scrollbar.config(command=self.tree.yview)

        self.tree.bind("<<TreeviewSelect>>", self.on_email_select)

        self.refresh_btn = tk.Button(
            self.receive_frame, text="Odswiez", command=self.refresh_in_background
        )
        self.refresh_btn.pack(pady=5)

        # message content
        tk.Label(self.receive_frame, text="Tresc wiadomosci:").pack(anchor="w")

        msg_frame = tk.Frame(self.receive_frame)
        msg_frame.pack(fill="both", expand=True, pady=5)

        self.msg_scrollbar = tk.Scrollbar(msg_frame)
        self.msg_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.message_view = tk.Text(
            msg_frame,
            height=8,
            width=78,
            state=tk.DISABLED,
            yscrollcommand=self.msg_scrollbar.set,
        )
        self.message_view.pack(side=tk.LEFT, fill="both", expand=True)
        self.msg_scrollbar.config(command=self.message_view.yview)

        # send section
        self.send_frame = tk.LabelFrame(
            self.root, text="Wyslij nowa wiadomosc", padx=10, pady=10
        )
        self.send_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(self.send_frame, text="Odbiorca:").grid(
            row=0, column=0, sticky="e", pady=2
        )
        self.to_entry = tk.Entry(self.send_frame, width=65)
        self.to_entry.grid(row=0, column=1, pady=2, sticky="w")

        tk.Label(self.send_frame, text="Temat:").grid(
            row=1, column=0, sticky="e", pady=2
        )
        self.subject_entry = tk.Entry(self.send_frame, width=65)
        self.subject_entry.grid(row=1, column=1, pady=2, sticky="w")

        tk.Label(self.send_frame, text="Tresc:").grid(
            row=2, column=0, sticky="ne", pady=2
        )
        self.body_text = tk.Text(self.send_frame, width=65, height=6)
        self.body_text.grid(row=2, column=1, pady=2)

        self.send_btn = tk.Button(
            self.send_frame, text="Wyslij wiadomosc", command=self.send_in_background
        )
        self.send_btn.grid(row=3, column=1, pady=10, sticky="e")

        # autoresponder
        self.auto_frame = tk.LabelFrame(
            self.root, text="Autoresponder", padx=10, pady=10
        )
        self.auto_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(self.auto_frame, text="Temat automatycznej odpowiedzi:").grid(
            row=0, column=0, sticky="e"
        )
        self.auto_subject = tk.Entry(self.auto_frame, width=40)
        self.auto_subject.insert(0, "Automatyczna odpowiedz: Aktualnie mnie nie ma")
        self.auto_subject.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        tk.Label(self.auto_frame, text="Tresc automatycznej odpowiedzi:").grid(
            row=1, column=0, sticky="ne"
        )
        self.auto_body = tk.Text(self.auto_frame, width=40, height=3)
        self.auto_body.insert("1.0", "Hej, aktualnie mnie nie ma. Odpowiem po powrocie")
        self.auto_body.grid(row=1, column=1, padx=5, pady=2)

        self.auto_btn = tk.Button(
            self.auto_frame,
            text="Uruchom Autoresponder",
            bg="lightgreen",
            command=self.toggle_autoresponder,
        )
        self.auto_btn.grid(row=0, column=2, rowspan=2, padx=15)

    def update_email_list(self, emails):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.current_emails = emails
        self.displayed_emails = emails

        if not emails:
            self.tree.insert("", tk.END, values=("Brak wiadomosci", ""))
            return

        for email in emails:
            tags = ("unseen",) if email.get("is_unseen") else ()
            self.tree.insert(
                "", tk.END, values=(email["sender"], email["subject"]), tags=tags
            )

    def fetch_task(self):
        emails = self.engine.fetch_emails(limit=20)
        self.root.after(0, self.update_email_list, emails)

    def refresh_in_background(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.insert("", tk.END, values=("Pobieranie wiadomosci...", ""))
        threading.Thread(target=self.fetch_task, daemon=True).start()

    def on_email_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        index = self.tree.index(item)

        if index < len(self.displayed_emails):
            selected_email = self.displayed_emails[index]

            current_tags = self.tree.item(item, "tags")
            if "unseen" in current_tags:
                self.tree.item(item, tags=())

            self.message_view.config(state=tk.NORMAL)
            self.message_view.delete("1.0", tk.END)
            self.message_view.insert(tk.END, selected_email.get("body", "Brak tresci."))
            self.message_view.config(state=tk.DISABLED)

    def send_in_background(self):
        to_address = self.to_entry.get()
        subject = self.subject_entry.get()
        body = self.body_text.get("1.0", tk.END).strip()

        if not to_address or not subject:
            messagebox.showwarning("Error", "Uzupelnij wszystkie pola")
            return

        threading.Thread(
            target=self.send_task, args=(to_address, subject, body), daemon=True
        ).start()

    def send_task(self, to_address, subject, body):
        success = self.engine.send_email(to_address, subject, body)
        if success:
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Sukces", "Wiadomosc wyslana z powodzeniem"
                ),
            )
            self.root.after(0, self.clear_form)
        else:
            self.root.after(
                0,
                lambda: messagebox.showerror("Blad", "Nie udalo sie wyslac wiadomosci"),
            )

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

    def apply_nlp_filter(self):
        keyword = self.filter_entry.get().strip()
        if not keyword:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.insert("", tk.END, values=(f"Filtrowanie NLP dla: {keyword}...", ""))

        threading.Thread(target=self._filter_task, args=(keyword,), daemon=True).start()

    def _filter_task(self, keyword):
        filtered_emails = self.engine.filter_emails_by_keyword(
            self.current_emails, keyword
        )
        self.root.after(0, self._show_filtered_results, filtered_emails)

    def _show_filtered_results(self, filtered_emails):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.displayed_emails = filtered_emails

        if not filtered_emails:
            self.tree.insert("", tk.END, values=("Brak dopasowan dla tego slowa", ""))
            return

        for email in filtered_emails:
            tags = ("unseen",) if email.get("is_unseen") else ()
            self.tree.insert(
                "", tk.END, values=(email["sender"], email["subject"]), tags=tags
            )

    def reset_filter(self):
        self.filter_entry.delete(0, tk.END)
        self.update_email_list(self.current_emails)


if __name__ == "__main__":
    root = tk.Tk()
    app = MailClientGUI(root)
    root.mainloop()
