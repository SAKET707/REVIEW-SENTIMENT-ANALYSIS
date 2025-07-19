import pandas as pd
import numpy as np
import joblib 
import re

import nltk
from nltk.data import find
try:
    find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab') 

try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


try:
    find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


from nltk.tokenize import sent_tokenize,word_tokenize
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

xgb = joblib.load('model.joblib')
tfidf_review = joblib.load('tfidf.joblib')

def predict(review):
    review=review.lower()

    def count_mojibake(review):
        if pd.isna(review):
            return 0
        return len(re.findall(r'[^\x00-\x7F]', review))
    
    mojibake_freq = count_mojibake(review)
    has_mojibake = 0 if mojibake_freq==0 else 1

    review_char = len(review)
    review_words = len(word_tokenize(review))
    review_sent = len(sent_tokenize(review))

    def rep_score(text):
        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return 0
        word_counts = Counter(words)
        repeated = sum([count for count in word_counts.values() if count > 1])
        return repeated / total_words

    repetition_score = rep_score(review)

    def max_word_freq(text):
        words = text.lower().split()
        if not words:
            return 0
        word_counts = Counter(words)
        return max(word_counts.values()) / len(words)

    max_word_repeat_ratio = max_word_freq(review)

    meaningful_stopwords={'none', 'believe', 'seem', 'nonetheless', 'than', 'always', 'whereas', 'when', "aren't", 'even though',
                       "needn't", 'now', 'any', "shouldn't", 'in case', 'felt', 'literally', 'finally', "doesn't",
                        'sorta', "isn't", 'yet', "don't", 'too', 'ever', 'know', 'eventually', 'while', 'most', 'rather', "weren't", 'even', 'recently',
                        'mean', 'why', 'but', "won't",
                        'so', 'no', 'instead', 'nowhere', 'all', 'anymore', "hasn't", 'several', 'worst', "didn't", 'soon',
                        'what', "couldn't", 'best', 'feel', 'many', 'worse', 'kinda', 'during', 'until', 'though', 'extremely', 'seems', 'honestly',
                        'think', 'although', 'how', 'where',
                        'who', "ain't", 'nothing', 'again', 'better', 'then', 'unless', 'before', 'barely', 'enough', 'because', 'never', 'since',
                        'hardly', 'assume', 'merely', 'which', "wouldn't", 'not',
                        'more', "mightn't", 'few', 'just', 'fewer', 'less', 'nevertheless', 'only', 'if', 'suppose', "can't", 'later', 'already',
                        "mustn't", 'absolutely',
                        'completely', 'expect', 'after', 'scarcely', 'however', 'nor', "hadn't", 'very', "haven't", 'consider', 'as',
                        'really', 'hopefully', 'somewhat', "shan't", 'still', "wasn't", 'understand', 'seemed', 'utterly', 'apparently'}

    stop_words = set(stopwords.words('english')) - meaningful_stopwords

    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)

    def transformtext(text):
        if pd.isna(text):
            return ""

        tokens = tokenizer.tokenize(text)
        clean_tokens = []

        for word in tokens:
            if word in stop_words:
                continue

            if re.match(r"^[a-zA-Z0-9']+$", word):
                clean_tokens.append(lemmatizer.lemmatize(word))
            elif re.match(r"^#\w+$", word):        
                clean_tokens.append(word)
            elif re.match(r"^@\w+$", word):        
                clean_tokens.append(word)
            elif word in {"!", "?"}:
                clean_tokens.append(word)

        return " ".join(clean_tokens)
    
    review = transformtext(review)

    
    def get_sentiment(review):
        analysis = TextBlob(review)
        return analysis.sentiment.polarity
    
    sentiment_score = get_sentiment(review)

    extra_features = np.array([
        mojibake_freq,
        has_mojibake,
        review_char,
        review_words,
        review_sent,
        repetition_score,
        max_word_repeat_ratio,
        sentiment_score
    ]).reshape(1, -1)

    extra_sparse = csr_matrix(extra_features)
    review_tfidf = tfidf_review.transform([review])

    final_input = hstack([review_tfidf, extra_sparse])

    prediction = xgb.predict(final_input)[0]

    return prediction
