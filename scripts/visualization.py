import pickle
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def display_topics(lda, vectorizer, no_top_words=10):
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        print(f"Topic {topic_idx}: {topics[topic_idx]}")
    return topics

def generate_word_clouds(topics):
    os.makedirs('results/', exist_ok=True)
    for topic_idx, words in topics.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for Topic {topic_idx}")
        plt.savefig(f'results/topic_{topic_idx}_wordcloud.png')
        plt.close()

def create_pyldavis_visualization(lda, doc_term_matrix, vectorizer):
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.prepare(lda, doc_term_matrix, vectorizer.get_feature_names_out(), mds='tsne')
    pyLDAvis.save_html(panel, 'results/lda_visualization.html')
    print("pyLDAvis visualization saved as lda_visualization.html")

if __name__ == "__main__":
    # Load model, vectorizer, and document-term matrix
    with open('models/lda_model.pkl', 'rb') as f:
        lda = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('data/processed/doc_term_matrix.pkl', 'rb') as f:
        doc_term_matrix = pickle.load(f)
    
    # Display topics
    topics = display_topics(lda, vectorizer)
    
    # Generate word clouds
    generate_word_clouds(topics)
    
    # Create and save pyLDAvis visualization
    create_pyldavis_visualization(lda, doc_term_matrix, vectorizer)



