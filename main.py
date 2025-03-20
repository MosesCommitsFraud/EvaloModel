import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle

# Load the dataset
df = pd.read_csv("balanced_feedback_dataset.csv")
print("Dataset preview:")
print(df.head())

# Map sentiment labels to integers
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['label'].map(label_mapping)

# Prepare texts and labels using the correct column names
texts = df['feedback'].tolist()
labels = df['label'].tolist()

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load the pre-trained BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define a custom Dataset class
class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove the extra batch dimension
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Create the PyTorch datasets
train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer)
val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer)

# Define a compute_metrics function to evaluate accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Set up training arguments with improvements
training_args = TrainingArguments(
    output_dir='./results',                # Where to store the final model
    num_train_epochs=3,                    # Maximum number of training epochs
    per_device_train_batch_size=16,        # Training batch size per device
    per_device_eval_batch_size=64,         # Evaluation batch size per device
    warmup_steps=500,                      # Warmup steps for the learning rate scheduler
    weight_decay=0.01,                     # Weight decay for optimization
    evaluation_strategy="epoch",           # Evaluate at the end of each epoch
    save_strategy="epoch",                 # Save checkpoint at the end of each epoch
    logging_dir='./logs',                  # Directory for logs
    logging_steps=10,
    load_best_model_at_end=True,           # Load the best model at the end of training
    fp16=True,                             # Enable mixed precision training
    report_to=None,                        # Disable logging integrations (like wandb) to reduce overhead
    lr_scheduler_type="cosine",            # Use a cosine learning rate scheduler
    learning_rate=2e-5,                    # Set a lower learning rate for fine-tuning
)

# Initialize the Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Start training
print("Starting training...")
trainer.train()

# Evaluate the model on the validation set and print the results
results = trainer.evaluate()
print("Evaluation results:", results)

# Save the trained model and tokenizer
output_dir = "./saved_model"
os.makedirs(output_dir, exist_ok=True)

print(f"Saving model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save label mapping for future use
with open(os.path.join(output_dir, "label_mapping.pkl"), "wb") as f:
    pickle.dump(label_mapping, f)

print("Model, tokenizer, and label mapping saved successfully!")

# Create a helper function for prediction (you can save this to a separate file)
def predict_sentiment(text, model_dir="./saved_model"):
    # Load the saved model, tokenizer, and label mapping
    loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
    loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)

    with open(os.path.join(model_dir, "label_mapping.pkl"), "rb") as f:
        loaded_label_mapping = pickle.load(f)

    # Create reverse mapping (from integers to labels)
    reverse_mapping = {v: k for k, v in loaded_label_mapping.items()}

    # Tokenize input text
    inputs = loaded_tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # Get prediction
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

    # Get confidence score and label
    confidence = predictions[0][predicted_class].item()
    predicted_label = reverse_mapping[predicted_class]

    return {
        "text": text,
        "label": predicted_label,
        "confidence": confidence,
        "all_scores": {reverse_mapping[i]: score.item() for i, score in enumerate(predictions[0])}
    }

# Example usage of prediction function:
# result = predict_sentiment("This product is amazing!")
# print(result)