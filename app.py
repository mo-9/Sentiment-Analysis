
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle


# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

@st.cache_data
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load LSTM model and tokenizer
@st.cache(allow_output_mutation=True)
def load_lstm_components(model_path: str, tokenizer_path: str):
    """Loads the model and tokenizer. Ensure the .h5 and .pickle files are in the app folder."""
    model = None
    tokenizer = None
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model from '{model_path}'. Place 'best_lstm_model.h5' in the app directory.\nError: {e}")
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load tokenizer from '{tokenizer_path}'. Place 'tokenizer.pickle' in the app directory.\nError: {e}")
    return model, tokenizer

# Constants
MODEL_PATH = 'best_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_SEQUENCE_LENGTH = 50

# Initialize components
model, tokenizer = load_lstm_components(MODEL_PATH, TOKENIZER_PATH)
if model is None or tokenizer is None:
    st.stop()

# Streamlit App
st.title("Tweet Sentiment Analysis with LSTM")
st.markdown("This app loads a pre-trained LSTM model to predict tweet sentiment")

user_input = st.text_area("Enter a tweet to analyze:", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.error("Please enter tweet text to analyze.")
    else:
        # Preprocess and encode
        processed = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([processed])
        padded_seq = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        # Predict
        preds = model.predict(padded_seq)
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = label_map.get(class_idx, 'unknown')
        # Display
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.2f}")