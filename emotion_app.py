import streamlit as st
import torch
import pandas as pd
from transformers import (
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    DistilBertTokenizer,
    AlbertTokenizer,
    RobertaTokenizer,
    BertTokenizer
)
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CombinedEmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        return item

def predict(model, texts, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataset = CombinedEmotionDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16)
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())
    
    return np.array(predictions)

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
    distilbert_model.load_state_dict(torch.load('backend/final_distilbert_model.pth', map_location=device))
    
    albert_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=5)
    albert_model.load_state_dict(torch.load('backend/final_albert_model.pth', map_location=device))
    
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
    roberta_model.load_state_dict(torch.load('backend/final_roberta_model.pth', map_location=device))
    
    final_bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    final_bert_model.load_state_dict(torch.load('backend/final_bert_model.pth', map_location=device))

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('backend/final_distilbert_tokenizer')
    albert_tokenizer = AlbertTokenizer.from_pretrained('backend/final_albert_tokenizer')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('backend/final_roberta_tokenizer')
    bert_tokenizer = BertTokenizer.from_pretrained('backend/final_bert_tokenizer')

    return (
        distilbert_model, albert_model, roberta_model, final_bert_model,
        distilbert_tokenizer, albert_tokenizer, roberta_tokenizer, bert_tokenizer
    )

def main():
    # Set page configuration
    st.set_page_config(page_title="Emotion Analysis App", layout="wide")

    # Custom CSS for gradient background and styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #000000, #434343);  /* Gradient from black to dark gray */
            color: white;  /* Change text color for better visibility */
        }
        .input-box {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .results-box {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            text-align: center;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Emotion Analysis App")

    # Load models
    (distilbert_model, albert_model, roberta_model, final_bert_model,
     distilbert_tokenizer, albert_tokenizer, roberta_tokenizer, bert_tokenizer) = load_models()

    # Text input
    with st.container():
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        text = st.text_area("Enter text for emotion analysis:", height=150)
        analyze_button = st.button("Analyze")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_button:
        if text:
            # Create DataFrame
            df = pd.DataFrame({'text': [text]})

            # Get predictions from each model
            distilbert_preds = predict(distilbert_model, df, distilbert_tokenizer)
            albert_preds = predict(albert_model, df, albert_tokenizer)
            roberta_preds = predict(roberta_model, df, roberta_tokenizer)

            # Add predictions to the DataFrame
            df['DistilBERT_Predictions'] = distilbert_preds.tolist()
            df['ALBERT_Predictions'] = albert_preds.tolist()

            # Get final predictions from BERT
            final_preds = predict(final_bert_model, df, bert_tokenizer)

            # Display results without percentage scores
            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.subheader("Emotion Analysis Results:")
            
            emotion_labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
            
            for emotion_label, score in zip(emotion_labels, final_preds[0]):
                if score > 0.5:  # Threshold for positive prediction (you can adjust this value)
                    st.write(f"{emotion_label}: Yes")
                else:
                    st.write(f"{emotion_label}: No")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text for analysis.")

    # Footer with copyright notice
    st.markdown(
        '<div class="footer">© 2024 Vedant Patil. All rights reserved.</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()