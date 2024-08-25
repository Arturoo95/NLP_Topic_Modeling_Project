import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

def train_lda_model(documents, n_topics=10):
    # Convert the processed documents into a bag-of-words representation
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    return lda, vectorizer, doc_term_matrix

def save_model(lda, vectorizer):
    os.makedirs('models/', exist_ok=True)
    with open('models/lda_model.pkl', 'wb') as f:
        pickle.dump(lda, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    # Load processed documents
    with open('data/processed/processed_docs.pkl', 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Train the LDA model
    lda, vectorizer, doc_term_matrix = train_lda_model(processed_docs)
    
    # Save the model and vectorizer
    save_model(lda, vectorizer)
    
    # Optionally save the document-term matrix
    with open('data/processed/doc_term_matrix.pkl', 'wb') as f:
        pickle.dump(doc_term_matrix, f)
    
    print("LDA model training complete. Model and vectorizer saved.")

