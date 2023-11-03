import torch
from transformers import BertTokenizer, BertForSequenceClassification

def sentiment_analysis(text):
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Tokenize and encode the input text
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)

    # Convert the inputs to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted sentiment label
    logits = outputs[0]
    predicted_label = torch.argmax(logits).item()

    # Map the predicted label to sentiment class
    sentiment_classes = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_classes[predicted_label]

    return sentiment
