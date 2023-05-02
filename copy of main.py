import time
import tkinter
import tkinterweb
import re
import sys
import webbrowser
from tkinter import Menu
from tkinter import simpledialog
from tkinter import *
from tkinter.ttk import *
from ttkthemes import ThemedTk
from tkinterhtml import HtmlFrame
from tkinter import TclError

from nlp import (get_response, update_bot_json, load_response_data,
                 is_greeting, is_farewell)

# Add this line at the beginning of main.py
print("Starting the Chatbot...")

# GUI settings
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
PLACEHOLDER_COLOR = "#777777"  # Gray color for the placeholder text

def send():
    user_input = e.get("1.0", "end-1c")
    if user_input == 'Type your message here...':
        return
    send = "You -> " + user_input
    txt.insert(END, "\n" + send)

    e.delete("1.0", END)

    # Typing indicator
    txt.insert(END, "\n" + "Bot -> " + "Typing...")
    root.update()
    time.sleep(1)  # Adjust the typing delay here
    txt.delete("end-2l", "end-1c")

    bot_response = get_response(user_input)
    txt.insert_html(f'<p>Bot -> {bot_response}</p>')

    txt.see(END)

def on_entry_click(event):
    """Clear the placeholder text when the input field is clicked."""
    if e.get("1.0", "end-1c") == 'Type your message here...':
        e.delete("1.0", "end")
        e.config(foreground=TEXT_COLOR)

def on_focusout(event):
    if e.get("1.0", "end-1c") == '':
        e.insert("1.0", 'Type your message here...')
        e.config(foreground=PLACEHOLDER_COLOR)

# Create the main application window with a theme
try:
    root = ThemedTk(theme="arc")
except TclError as e:
    print("Error initializing the GUI:", e)
    sys.exit(1)
root.title("Chatbot")

lable1 = Label(root, text="Welcome to PCS Assistant", font=FONT_BOLD, width=80)
lable1.grid(row=0, pady=10)

chat_frame = tkinter.Frame(root, bg=BG_COLOR)
chat_frame.grid(row=1, column=0, columnspan=2)

txt = HtmlFrame(chat_frame, width=70)
txt.pack(fill=BOTH, expand=True)
txt.set_content("<p>&nbsp;</p>")  # Add a non-breaking space as the initial text
txt.grid(row=1, column=0, columnspan=2)
css = f"""
    body {{
        color: {TEXT_COLOR};
        font-family: {FONT.split()[0]};
        font-size: {FONT.split()[1]}pt;
        white-space: pre-wrap;
    }}
"""
txt.html("<style>" + css + "</style>")

scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

e = Text(root, height=2, font=FONT, width=55)
e.insert(1.0, 'Type your message here...')
e.grid(row=2, column=0)
e.bind('<FocusIn>', on_entry_click)
e.bind('<FocusOut>', on_focusout)

send_button = Button(root, text="Send", style='Custom.TButton', command=send)
send_button.grid(row=2, column=1)

root.bind('<Return>', lambda event=None: send())

# Autofocus on the input field
e.focus

root.iconphoto(False, PhotoImage(file='icon.png'))

# Create a menu bar
menu_bar = Menu(root)
root.config(menu=menu_bar)

# Create a function to handle adding keywords and responses
def add_keywords_and_responses():
    keyword = simpledialog.askstring("Add Keyword", "Enter the keyword:")
    response = simpledialog.askstring("Add Response", "Enter the response (add 'http://' or 'https://' for hyperlinks):")

    if keyword and response:
        # Preserve hyperlinks in the response text
        response = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', response)
        update_bot_json(keyword, keyword, response)
        load_response_data("bot.json")  # Replace 'your_filename_here.json' with the appropriate filename.  # Reload response data in the chatbot window
        
# Add a drop-down menu for adding keywords and responses
add_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Options", menu=add_menu)
add_menu.add_command(label="Add Keywords & Responses", command=add_keywords_and_responses)

# Add this line just before root.mainloop()
print("Launching the GUI...")

root.mainloop()