import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

def extract_keywords(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize an empty list to store the keywords
    keywords = []
    
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)
        
        # Remove stopwords and punctuation
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
        
        # Perform part-of-speech tagging
        pos_tags = pos_tag(words)
        
        # Extract named entities
        named_entities = ne_chunk(pos_tags)
        
        # Iterate over the named entities and add them to the keywords list
        for entity in named_entities:
            if hasattr(entity, 'label') and entity.label() == 'NE':
                keywords.append(' '.join(c[0] for c in entity))
    
    return keywords
