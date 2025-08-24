import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from transformers import pipeline
import numpy as np
# print("numpy version is " + np.__version__)

# Download necessary NLTK datasets
nltk.download('punkt')

# Sample text
document = "Natural language processing (NLP) is a field of AI that helps machines understand human language. NLP is used in various applications like sentiment analysis, machine translation, and text summarization."

# Tokenization
words = word_tokenize(document)
sentences = sent_tokenize(document)
print("Word Tokenization:", words)
print("Sentence Tokenization:", sentences)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([document])
print("TF-IDF Features:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:", tfidf_matrix.toarray())

# Sentiment Analysis
blob = TextBlob(document)
print("Sentiment Polarity:", blob.sentiment.polarity)

# Named Entity Recognition (NER)

# Load the NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# Sample text
sample_text = "Sundar Pichai, the CEO of Google, announced a partnership with Microsoft in California."
# Perform NER
ner_results = ner_pipeline(sample_text)

# Print named entities
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']:.4f}")


# Text Summarization
summarizer = pipeline("summarization")
summarized_text = summarizer(document, max_length=50, min_length=25, do_sample=False)
print("Summarized Text:", summarized_text[0]['summary_text'])

# Machine Translation (English to French)
# translator = pipeline("translation_en_to_fr")
# Load the English-to-Chinese translation model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
translated_text = translator(document)
print("Translated Text ():", translated_text[0]['translation_text'])

