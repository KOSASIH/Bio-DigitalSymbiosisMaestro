import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(doc1, doc2):
    # Tokenize the documents into words
    nltk.download('punkt')
    doc1_tokens = nltk.word_tokenize(doc1)
    doc2_tokens = nltk.word_tokenize(doc2)
    
    # Combine the tokens into sentences
    doc1_sentence = ' '.join(doc1_tokens)
    doc2_sentence = ' '.join(doc2_tokens)
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([doc1_sentence, doc2_sentence])
    
    # Calculate the cosine similarity between the vectors
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return similarity_score

# Example usage
document1 = "This is the first document."
document2 = "This document is the second document."
similarity = calculate_similarity(document1, document2)
print(f"The similarity between the two documents is: {similarity}")
