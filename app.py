from tkinter import *
from main import bot_name, get_response
import logging

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    """
    Graphical interface for the Chatbot application.
    """
    def __init__(self) -> None:
        """
        Initializes the ChatApplication class.
        """
        self.window = Tk()
        self.history = []
        self._setup_main_window()

    def run(self):
        """
        Runs the Tkinter main loop.
        """
        self.window.mainloop()

    def _setup_main_window(self):
        """
        Sets up the main window and its components.
        """
        self.window.title("Chat")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg=BG_COLOR)

        # Head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # Tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # Bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # Message entry
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # Send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

        # Bind Escape key to exit fullscreen
        self.window.bind("<Escape>", self._exit_fullscreen)

    def _on_enter_pressed(self, event):
        """
        Inserts a message when Enter key is pressed.
        """
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        self.history.append((msg, "You"))
        response = get_response(msg)
        self._insert_message(response, bot_name)
        self.history.append((response, bot_name))

    def _insert_message(self, msg, sender):
        """
        Inserts a message into the text widget.
        """
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def _exit_fullscreen(self, event):
        """
        Exits fullscreen mode when Escape key is pressed.
        """
        self.window.attributes('-fullscreen', False)
        self._save_chat_history()

    def _save_chat_history(self):
        """
        Saves the chat history to a text file.
        """
        try:
            with open("chat_history.txt", "w") as fin:
                for msg, sender in self.history:
                    fin.write(f"{sender}: {msg}\n")
            logging.info("Chat history saved successfully.")
        except OSError as err:
            logging.error(f"Error saving chat history: {err}")


if __name__ == "__main__":
    logging.basicConfig(filename='chat_application.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        app = ChatApplication()
        app.run()
    except RuntimeError as err:
        logging.error(f"Error initializing or running ChatApplication: {err}")
