# topics
import nltk
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    "Machine learning and AI are transforming the world of technology.",
    "Elon Musk and Tesla are leading the electric vehicle industry.",
    "Bitcoin and Ethereum are revolutionizing the financial sector.",
    "Python and deep learning frameworks like TensorFlow are popular for AI development."
]

# Preprocess text: Tokenization & Remove stopwords
"""
Converts each document to lowercase, tokenizes it into words, 
removes punctuation, and filters out common English stopwords (like "the", "and", etc.)
"""
stop_words = set(stopwords.words('english'))
tokenized_docs = [[word.lower() for word in word_tokenize(doc) if word.isalpha() and word.lower() not in stop_words] for doc in documents]

# Create a dictionary and corpus
"""
Builds a Gensim dictionary mapping each unique word to an ID.
Converts each document into a bag-of-words (BoW) representation (a list of (word_id, count) pairs).
"""
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

# Train LDA model with 2 topics
"""
Trains a Latent Dirichlet Allocation (LDA) model to find 2 topics in the corpus.
"""
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# Print topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

