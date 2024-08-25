import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
import pickle
import os

# Ensure NLTK data packages are downloaded

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text using RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Remove stop words and lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join the tokens back into a string
    return ' '.join(tokens)

def preprocess_documents(documents):
    return [preprocess_text(doc) for doc in documents]

if __name__ == "__main__":
    # Assume we're loading the raw data from a file or directly in code
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    processed_docs = preprocess_documents(newsgroups_data.data)
    
    # Save the processed documents
    os.makedirs('data/processed/', exist_ok=True)
    with open('data/processed/processed_docs.pkl', 'wb') as f:
        pickle.dump(processed_docs, f)
    
    print("Data preprocessing complete. Processed documents saved.")

