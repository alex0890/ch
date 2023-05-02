import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from transformers import Trainer, TrainingArguments

# Load your dataset from the text file
data = pd.read_csv('your_dataset.csv') # Replace with the path to your dataset file
train_texts = data['text'].tolist()
train_labels = data['label'].tolist() # Replace with the appropriate column name for labels

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize the input texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Prepare the dataset for training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)

# Configure the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Create the Trainer and start training
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine-tuned-bert')

