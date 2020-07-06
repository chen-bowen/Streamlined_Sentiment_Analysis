from sklearn.base import BaseEstimator
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.utils.word_tokenizer import WordTokenizer
from collections import Counter
import pandas as pd
import numpy as np


class TermFrequency_InvDocFrequency(BaseEstimator):
    """ Custom TF-IDF transformer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = WordTokenizer()

    @staticmethod
    def get_document_frequency(words_list, document_list):
        """ obtain the document frequency of each word """
        # calculate num of existences for each unique words
        document_frequency = {
            word: sum([word in document for document in document_list])
            for word in words_list
        }
        return document_frequency

    def fit(self, X, y=None):
        """ save inverse document frequency """
        # get tokenized document
        tokenized_documents = [self.tokenizer.tokenize_sentence(s) for s in X]
        # get unique words
        self.all_unique_words = list(set(chain(*tokenized_documents)))
        # idf = log(N/(1 + df))
        self.num_documents = len(tokenized_documents)
        document_frequency = self.get_document_frequency(
            self.all_unique_words, tokenized_documents
        )
        self.idfs = {
            word: np.log(self.num_documents / (doc_freq + 1))
            for word, doc_freq in document_frequency.items()
        }
        return self

    def transform(self, X):
        """ Transform the input X"""
        # find term frequencies
        occurrences_count = list(map(Counter, X))
        document_lengths = list(map(len, X))

        tfs = [
            {word: count / length for word, count in counts}
            for counts, length in zip(occurrences_count, document_lengths)
        ]

        # find tf_idf

