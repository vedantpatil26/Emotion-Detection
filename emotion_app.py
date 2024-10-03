import streamlit as st
import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    AlbertTokenizer, 
    AlbertForSequenceClassification, 
    RobertaTokenizer, 
    RobertaForSequenceClassification
)

selected_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

class EnsembleModel(nn.Module):
    def __init__(self, num_labels, num_models):
        super(EnsembleModel, self).__init__()
        self.classifier = nn.Linear(num_labels * num_models, num_labels)
        
    def forward(self, model_outputs):
        logits = self.classifier(model_outputs)
        return logits

def ensemble_predict(ensemble_model, models, tokenizers, texts, device, max_len=128):
    ensemble_model.eval()
    for model in models:
        model.eval()
    
    with torch.no_grad():
        model_outputs = []
        for model, tokenizer in zip(models, tokenizers):
            encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            model_outputs.append(torch.sigmoid(logits))
        
        model_outputs = torch.cat(model_outputs, dim=1)
        ensemble_logits = ensemble_model(model_outputs)
        probs = torch.sigmoid(ensemble_logits).cpu().numpy()
    
    return probs

def load_models(sequential=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if sequential:
        distilbert_path = './backend/final_distilbert_model.pth'
        albert_path = './backend/final_albert_model.pth'
        roberta_path = './backend/final_roberta_model.pth'
        
        distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(selected_emotions)).to(device)
        albert_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(selected_emotions)).to(device)
        roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(selected_emotions)).to(device)

        distilbert_model.load_state_dict(torch.load(distilbert_path))
        albert_model.load_state_dict(torch.load(albert_path))
        roberta_model.load_state_dict(torch.load(roberta_path))
        
    else:
        distilbert_path = './distilbert_emotion_model'
        albert_path = './albert_emotion_model'
        roberta_path = './roberta_emotion_model'
    
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path).to(device)
        albert_model = AlbertForSequenceClassification.from_pretrained(albert_path).to(device)
        roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_path).to(device)

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path if not sequential else 'distilbert-base-uncased')
    albert_tokenizer = AlbertTokenizer.from_pretrained(albert_path if not sequential else 'albert-base-v2')
    roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path if not sequential else 'roberta-base')
    
    return [distilbert_model, albert_model, roberta_model], [distilbert_tokenizer, albert_tokenizer, roberta_tokenizer]

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    html, body {
        background: linear-gradient(to right, #000000, #434343);
        color: white;
        margin: 0;
        padding: 0;
        height: 100%;
    }
    .main {
        flex: 1;
        padding: 20px;
        min-height: calc(100vh - 50px);
    }
    .footer {
        background: white;
        color: #434343;
        padding: 15px;
        text-align: center;
        width: 100%;
        position: fixed;
        bottom: 0;
        left: 0;
        margin: 0;
    }
    .footer-container {
        max-width: 1000px;
        margin: 0 auto;
    }
    .stTextInput, .stTextArea, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1);
        border: none;
        border-radius: 5px;
        color: white;
    }
    .stButton {
        background-color: #434343;
        color: white;
    }
    .stButton:hover {
        background-color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Emotion Classification with Ensemble Models")

model_choice = st.radio("Select Model Type:", options=["Parallel", "Sequential"])
sequential = model_choice == "Sequential"

st.write("Loading models...")
models, tokenizers = load_models(sequential)
ensemble_model = EnsembleModel(len(selected_emotions), 3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

st.success("Models loaded successfully!")

input_method = st.radio("Input Method:", ["Predefined Text", "Custom Text"])

if input_method == "Predefined Text":
    test_texts = [
        "The rollercoaster of emotions I experienced during the movie's climax left me breathless.",
        "As I stood atop the mountain, I felt exhilaration and trepidation.",
        "The unexpected news of my promotion filled me with joy and anxiety.",
        "Watching the sunset over the ocean brought me peace and melancholy.",
        "The heated argument with my best friend left me feeling anger and regret."
    ]
    selected_text = st.selectbox("Choose a text:", test_texts)
else:
    selected_text = st.text_area("Enter your text here:", "Type something...")

if st.button("Predict"):
    if selected_text.strip():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = ensemble_predict(ensemble_model, models, tokenizers, [selected_text], device)

        st.write(f"**Predictions for the text:**")
        st.write(selected_text)

        st.write("### Emotion Predictions:")
        for emotion, prob in zip(selected_emotions, predictions[0]):
            st.write(f"{emotion}: {prob:.4f} ({'Yes' if prob > 0.5 else 'No'})")
    else:
        st.warning("Please enter some text or choose a predefined option.")

st.markdown(
    '<div class="footer-container"><div class="footer">© 2024 Vedant Patil. All rights reserved.</div></div>', 
    unsafe_allow_html=True
)
