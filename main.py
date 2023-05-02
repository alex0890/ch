import sys
import time
import re
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QFontDialog
from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextBrowser, QPushButton, QVBoxLayout,
                             QWidget, QMenu, QAction, QInputDialog, QLineEdit, QLabel,
                             QListView, QPlainTextEdit)

from nlp import (get_response, update_bot_json, load_response_data,
                 is_greeting, is_farewell)

# Load pre-trained BERT tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(device)

def generate_answer(question, context):
    max_chunk_length = 350
    stride = 150

    # Tokenize the input text
    encoded_inputs = tokenizer.encode_plus(
        question, context, return_tensors='pt', max_length=max_chunk_length, truncation=True, stride=stride,
        padding="max_length"
    )
    input_ids_windows = encoded_inputs["input_ids"].split(max_chunk_length, dim=1)
    attention_mask_windows = encoded_inputs["attention_mask"].split(max_chunk_length, dim=1)
    inputs_list = [{"input_ids": ids.to(device), "attention_mask": mask.to(device)} for ids, mask in
                   zip(input_ids_windows, attention_mask_windows)]

    best_answer = None
    best_score = -1
    best_context = None

    for inputs in inputs_list:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Find the start and end positions of the answer in the context
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        try:
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            answer_score = (start_scores[0][answer_start] + end_scores[0][answer_end - 1]).item()

            if answer_score > best_score:
                best_score = answer_score
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
                context_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                context_start = max(answer_start - 50, 0)
                context_end = min(answer_end + 50, len(context_tokens))
                best_context = tokenizer.convert_tokens_to_string(context_tokens[context_start:context_end])
                best_answer = answer
        except:
            pass

    if best_answer is None:
        return "I'm sorry, I don't know the answer to that.", context
    else:
        return best_answer, best_context

# Load the context from a file
with open("context.txt", "r") as f:
    context = f.read()

# GUI settings
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class Chatbot(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Chatbot, self).__init__(*args, **kwargs)
        
        self.init_ui()
        self.chatbot_mode = "normal"
        self.custom_context = ""
        
    def init_ui(self):
        self.setWindowTitle("Chatbot")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.label = QLabel("Welcome to PCS Assistant")
        layout.addWidget(self.label)

        self.text_area = QTextBrowser()
        self.text_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_area.setOpenExternalLinks(True)
        layout.addWidget(self.text_area)

        self.input_field = QPlainTextEdit()
        layout.addWidget(self.input_field)
        self.input_field.installEventFilter(self)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send)
        layout.addWidget(self.send_button)

        central_widget.setLayout(layout)

        self.create_menu()

        self.text_area.installEventFilter(self)  # Add this line

        self.show()
        
    def show_help(self):
        help_text = ("To add or update keywords and responses, use the 'Options' menu and choose 'Add Keywords & Responses'.\n\n"
                 "To add patterns, you can modify the 'bot.json' file directly. Add an entry in the 'patterns' section with the "
                 "pattern as the key and the keyword as the value. After updating the 'bot.json' file, restart the application "
                 "for the changes to take effect.")
        QMessageBox.information(self, "Help", help_text)
        
    def load_custom_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Custom Dataset File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, "r") as f:
                self.custom_dataset = f.read()
            QMessageBox.information(self, "Success", "Custom dataset loaded successfully.")
        

    def eventFilter(self, source, event):
        if (event.type() == QEvent.KeyPress and source is self.input_field):
            if event.key() in (Qt.Key_Enter, Qt.Key_Return):
                self.send()
                return True
        return super().eventFilter(source, event)
    
    def save_chat_history(self):
        with open("chat_history.txt", "w") as chat_history_file:
            chat_history_file.write(self.text_area.toPlainText())

    def load_chat_history(self):
        try:
            with open("chat_history.txt", "r") as chat_history_file:
                chat_history = chat_history_file.read()
                self.text_area.setPlainText(chat_history)
        except FileNotFoundError:
            pass
        
    def save_chat_history(self):
        with open("chat_history.txt", "w") as chat_history_file:
            chat_history_file.write(self.text_area.toPlainText())

    def load_chat_history(self):
        try:
            with open("chat_history.txt", "r") as chat_history_file:
                chat_history = chat_history_file.read()
                self.text_area.setPlainText(chat_history)
        except FileNotFoundError:
            pass
        
    def set_font(self):
        current_font = self.text_area.currentFont()
        font, ok = QFontDialog.getFont(current_font, self)
        if ok:
            self.text_area.setCurrentFont(font)
            self.input_field.setCurrentFont(font)
            
    def set_color(self):
        current_color = self.text_area.textColor()
        color = QColorDialog.getColor(current_color, self)
        if color.isValid():
            self.text_area.setTextColor(color)

            palette = self.input_field.palette()
            palette.setColor(QPalette.Text, color)
            self.input_field.setPalette(palette)



    def create_menu(self):
        menu_bar = self.menuBar()
        
        file_menu = QMenu("&File", self)  
        menu_bar.addMenu(file_menu) 

        load_action = QAction("Load Chat History", self)
        load_action.triggered.connect(self.load_chat_history)
        file_menu.addAction(load_action)

        save_action = QAction("Save Chat History", self)
        save_action.triggered.connect(self.save_chat_history)
        file_menu.addAction(save_action)
        
        settings_menu = QMenu("&Settings", self)  # Add this line
        menu_bar.addMenu(settings_menu)
        
        font_action = QAction("Font", self)
        font_action.triggered.connect(self.set_font)
        settings_menu.addAction(font_action)

        color_action = QAction("Color", self)
        color_action.triggered.connect(self.set_color)
        settings_menu.addAction(color_action)

        options_menu = QMenu("&Options", self)
        menu_bar.addMenu(options_menu)

        add_kw_resp_action = QAction("Add Keywords & Responses", self)
        add_kw_resp_action.triggered.connect(self.add_keywords_and_responses)
        options_menu.addAction(add_kw_resp_action)

        chatbot_mode_menu = QMenu("&Chatbot Mode", self)
        menu_bar.addMenu(chatbot_mode_menu)

        normal_chatbot_action = QAction("Normal Chatbot", self)
        normal_chatbot_action.triggered.connect(lambda: self.set_chatbot_mode("normal"))
        chatbot_mode_menu.addAction(normal_chatbot_action)

        bert_chatbot_action = QAction("BERT QA Chatbot", self)
        bert_chatbot_action.triggered.connect(lambda: self.set_chatbot_mode("bert_qa"))
        chatbot_mode_menu.addAction(bert_chatbot_action)
        
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        options_menu.addAction(help_action)
        
        load_custom_dataset_action = QAction("Load Custom Dataset", self)
        load_custom_dataset_action.triggered.connect(self.load_custom_dataset)
        options_menu.addAction(load_custom_dataset_action)

    def send(self):
        user_input = self.input_field.toPlainText()
        if user_input.strip() == '':
            return

        self.text_area.append("You -> " + user_input)
        self.input_field.clear()

        QTimer.singleShot(1000, lambda: self.type_bot_response(user_input, "Bot -> Typing..."))

    def type_bot_response(self, user_input, typing_text="Bot -> Typing..."):
        self.text_area.undo()
        if self.chatbot_mode == "normal":
            bot_response = get_response(user_input)
        else:
            context_to_use = self.custom_context if self.custom_context else context  # Add this line
            bot_response, _ = generate_answer(user_input, context_to_use)  # Modify this line
        bot_response = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', bot_response)
        self.text_area.insertHtml('<br><span style="font-weight: bold; color: blue;">Bot -> </span>' + bot_response + '<br><br>')


    def set_chatbot_mode(self, mode):
        self.chatbot_mode = mode

    def add_keywords_and_responses(self):
        keyword, ok = QInputDialog.getText(self, "Add Keyword", "Enter the keyword:")
        if ok:
            response, ok = QInputDialog.getText(self, "Add Response", "Enter the response (add 'http://' or 'https://' for hyperlinks):")
            if ok:
                update_bot_json(keyword, keyword, response)
                load_response_data("bot.json")

def main():
    app = QApplication(sys.argv)
    chatbot = Chatbot()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

