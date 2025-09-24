import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and tokenizer
model = load_model('rumor_cnn_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_rumor(tweet):
    cleaned = preprocess_text(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    return 'Rumor' if pred > 0.5 else 'Non-Rumor'

# Streamlit App
st.title('Rumor Identification System')
tweet = st.text_input('Enter a Tweet:')
if st.button('Predict'):
    if tweet:
        result = predict_rumor(tweet)
        st.write(f'The tweet is classified as: **{result}**')
    else:
        st.write('Please enter a tweet.')