# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:54:53 2019

@author: Ashok
"""

import re
import numpy as np
import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity

english_stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = r"(?u)\b\w\w+\b"
stopwords = set(nltk.corpus.stopwords.words('english'))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=True,
                    stem=True):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE )#| re.LOCALE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed


def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def cosine_sim(x, y):
    try:
        if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
        if type(y) is np.ndarray: y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print (x)
        print (y)
        d = 0.
    return d