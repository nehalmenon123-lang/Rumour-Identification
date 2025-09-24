import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('dataset.csv')  # Adjust path/columns as needed
df['text'] = df['text'].apply(preprocess_text)
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF for traditional models
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train and evaluate traditional models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Tokenizer for DL models
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

# CNN Model
cnn_model = Sequential()
cnn_model.add(Embedding(5000, 128, input_length=100))
cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(MaxPooling1D(5))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_seq, y_test)
print(f"CNN Accuracy: {cnn_acc}")

# LSTM Model
lstm_model = Sequential()
lstm_model.add(Embedding(5000, 128, input_length=100))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_seq, y_test)
print(f"LSTM Accuracy: {lstm_acc}")

# Save best model (CNN)
cnn_model.save('rumor_cnn_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Confusion Matrix for CNN
y_pred_cnn = (cnn_model.predict(X_test_seq) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_cnn)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('CNN Confusion Matrix')
plt.savefig('cnn_confusion_matrix.png')

# Accuracy Comparison Plot
accuracies = [accuracy_score(y_test, models['Logistic Regression'].predict(X_test_tfidf)),
              accuracy_score(y_test, models['Naive Bayes'].predict(X_test_tfidf)),
              accuracy_score(y_test, models['Random Forest'].predict(X_test_tfidf)),
              accuracy_score(y_test, models['SVM'].predict(X_test_tfidf)),
              lstm_acc, cnn_acc]
model_names = ['LR', 'NB', 'RF', 'SVM', 'LSTM', 'CNN']
plt.bar(model_names, accuracies)
plt.title('Model Accuracy Comparison')
plt.savefig('accuracy_comparison.png')