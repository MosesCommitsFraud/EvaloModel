import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import os

def analyze_sentiment(user_input, model_dir="./saved_model", verbose=False):
    """
    Analyze user feedback to determine sentiment (positive, negative, or neutral).

    Args:
        user_input (str): The user's feedback text
        model_dir (str): Directory where the model is saved
        verbose (bool): Whether to print detailed information

    Returns:
        dict: Analysis results including sentiment
    """
    try:
        # Load the saved model, tokenizer, and label mapping
        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)

        with open(os.path.join(model_dir, "label_mapping.pkl"), "rb") as f:
            label_mapping = pickle.load(f)

        # Create reverse mapping (from integers to labels)
        reverse_mapping = {v: k for k, v in label_mapping.items()}

        # Tokenize input text
        inputs = tokenizer(
            user_input,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        # Get sentiment prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()

        # Get confidence score and sentiment label
        confidence = predictions[0][predicted_class].item()
        predicted_sentiment = reverse_mapping[predicted_class]

        # Get all sentiment scores
        all_scores = {reverse_mapping[i]: score.item() for i, score in enumerate(predictions[0])}

        # Prepare results
        results = {
            "text": user_input,
            "sentiment": predicted_sentiment,
            "confidence": round(confidence * 100, 1),
            "detailed_scores": {k: round(v * 100, 1) for k, v in all_scores.items()} if verbose else None
        }

        return results

    except Exception as e:
        return {
            "error": str(e),
            "text": user_input,
            "sentiment": "error"
        }

def interactive_sentiment_analyzer(model_dir="./saved_model"):
    """
    Interactive command-line interface for analyzing sentiment in user feedback.
    """
    print("\n===== Sentiment Analyzer =====")
    print("Type your text below, or 'quit' to exit.")

    while True:
        print("\n")
        user_input = input("Your text: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using Sentiment Analyzer!")
            break

        if not user_input.strip():
            print("Please enter some text.")
            continue

        # Analyze the sentiment
        result = analyze_sentiment(user_input, model_dir, verbose=True)

        if "error" in result:
            print(f"Error: {result['error']}")
            continue

        # Display results
        print("\n----- Analysis Results -----")
        print(f"Sentiment: {result['sentiment'].upper()} ({result['confidence']}% confidence)")

        # Show detailed sentiment scores
        if result['detailed_scores']:
            print("\nDetailed Sentiment Scores:")
            for sentiment, score in result['detailed_scores'].items():
                print(f"  {sentiment.capitalize()}: {score}%")

        print("--------------------------")

# Example usage in another script:
"""
from sentiment_analyzer import analyze_sentiment

# Analyze a single piece of text
feedback = "I really enjoyed using this product!"
result = analyze_sentiment(feedback)
print(f"Sentiment: {result['sentiment']}")
"""

# If running as a script, start the interactive interface
if __name__ == "__main__":
    model_directory = "./saved_model"  # Change this to your model directory if different
    interactive_sentiment_analyzer(model_directory)