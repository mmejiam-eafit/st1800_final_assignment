import numpy as np
# from textblob import TextBlob
import nltk
import re

from nltk import word_tokenize

from nltk.stem.porter import PorterStemmer
threshold = 0.1


# def getSentimentAnalysis(string_list):
# #     sentiment_analysis = [{}]*len(string_list)
#     polarity = np.zeros(len(string_list))
#     subjectivity = np.zeros(len(string_list))
#     for i, string in enumerate(string_list):
#         blob = TextBlob(string)
#         polarity[i] = blob.sentiment.polarity
#         subjectivity[i] = blob.sentiment.subjectivity

#     return (polarity, subjectivity)


# def getSentimentLabels(polarity):
#     if np.abs(polarity) <= threshold:
#         return 'NEUTRAL'
    
#     if polarity > threshold:
#         return 'POSITIVE'
    
#     if polarity < -1*threshold:
#         return 'NEGATIVE'
    
   


class CustomTokenizer(object):

    def __init__(self):
        self._porterStemmer = PorterStemmer()


    def __call__(self, doc):
#         print(',')
        return [self._porterStemmer.stem(w) for w in word_tokenize(doc)]
    
class CustomPreprocessor(object):
        
    def __call__(self, doc):
#         print('.')
        doc = doc.lower()
        doc_tokens = doc.split()
        doc_tokens = [re.sub('[^A-Za-z0-9]+', '', w) for w in doc_tokens]
        doc = ' '.join(doc_tokens)
        
        return doc