🚀 LSTM Sentiment Analysis Web App with Streamlit
This project is an interactive web application built using Streamlit that allows users to perform sentiment analysis on input text using a pre-trained LSTM (Long Short-Term Memory) model. It demonstrates an end-to-end deployment pipeline for a natural language processing (NLP) model trained in Keras and saved as best_lstm_model.h5, along with its associated tokenizer.

🔍 Project Features
✅ Real-time Sentiment Prediction using a Keras-based LSTM model.

✅ Streamlit Interface for easy user interaction and deployment.

✅ Tokenizer Loading with support for dynamic upload.

✅ Handles missing model/tokenizer errors gracefully.

✅ Supports deployment on local or cloud (Streamlit Cloud / Hugging Face Spaces) environments.

✅ Easily adaptable for other sequence models or NLP tasks.

🧠 Model Details
Architecture: LSTM-based binary classification

Framework: TensorFlow / Keras

Training Data: Cleaned and preprocessed sentiment-labeled text dataset (e.g., IMDb, Twitter)

Input Format: Tokenized padded sequences

Output: Binary classification — Positive or Negative sentiment

🗂️ Project Structure
bash
Copy
Edit
Sentiment_Analysis_App/
│
├── app.py                  # Main Streamlit app
├── best_lstm_model.h5      # Pre-trained LSTM model (optional: upload via sidebar)
├── tokenizer.pickle        # Tokenizer used for input preprocessing (optional: upload via sidebar)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── assets/                 # Optional: images, logos, examples





📦 Installation & Running
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/mo-9/sentiment-analysis.git
cd sentiment-analysis-app
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Streamlit App
bash
Copy
Edit
streamlit run app.py
4. Usage
Enter text in the input field

View predicted sentiment immediately

Upload your own best_lstm_model.h5 or tokenizer.pickle if not provided

💡 Example

Type something like "I love this movie!" and get an instant sentiment prediction.

📁 Requirements
Python 3.11+

Streamlit

TensorFlow / Keras

NumPy

Pickle

☁️ Deploying on Streamlit Cloud
Push the repo to GitHub

Go to streamlit.io/cloud

Create a new app and link your repo

Add best_lstm_model.h5 and tokenizer.pickle to the repo or upload them via the sidebar

📌 To-Do
 Add multi-class sentiment support

 Integrate attention mechanism visualization

 Switch to HuggingFace Transformers for BERT-based model

 Add logging and analytics

🙌 Acknowledgements
Keras LSTM examples

Streamlit documentation

NLP preprocessing guides

