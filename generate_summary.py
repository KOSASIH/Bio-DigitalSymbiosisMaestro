import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

def generate_summary(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stop words and perform stemming on the words
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    # Calculate the word frequency
    word_frequency = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            word = stemmer.stem(word.lower())
            if word not in stop_words:
                if word not in word_frequency.keys():
                    word_frequency[word] = 1
                else:
                    word_frequency[word] += 1
    
    # Calculate the sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        sentence_score = 0
        words = word_tokenize(sentence)
        for word in words:
            word = stemmer.stem(word.lower())
            if word in word_frequency.keys():
                sentence_score += word_frequency[word]
        sentence_scores[sentence] = sentence_score
    
    # Sort the sentences based on their scores in descending order
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate the summary by selecting the top sentences
    summary = ""
    num_sentences = min(3, len(sorted_sentences))  # Adjust the number of sentences in the summary as needed
    for i in range(num_sentences):
        summary += sorted_sentences[i][0] + " "
    
    return summary

# Example usage
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed at metus vitae dolor bibendum vestibulum. In ut nisl at est efficitur efficitur. Vivamus sit amet finibus metus. Nulla sed risus euismod, feugiat elit eget, interdum arcu. Sed id ornare elit. Sed cursus ultricies nisl, sit amet condimentum ex. Proin id fringilla lorem. Curabitur vestibulum, mi a tristique lacinia, est nibh dictum ex, nec sollicitudin tellus magna non purus. Nullam id ex sed tellus lacinia tincidunt. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Fusce vulputate, ex sed commodo finibus, elit quam semper quam, sed tristique nisl metus nec sapien."

summary = generate_summary(text)
print(summary)
