from sklearn.base import BaseEstimator
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.utils.word_tokenizer import WordTokenizer
from collections import Counter
import pandas as pd
import numpy as np


class WordFrequencyVectorizer(BaseEstimator):
    """ Generate features in word frequency vectors, used for pipeline """

    def __init__(self, **kwargs):
        self.tokenizer = WordTokenizer()
        super().__init__(**kwargs)

    def set_word_index_mapping(self):
        processed_review = ReviewProcessor()
        self.word_to_index_map = processed_review.word_to_index_map
        self.vocab_size = processed_review.vocab_size

    def get_word_frequency_vector(self, review_text):
        """ Get the word frequency vector for one tokenized review"""
        # get the tokenized review from the review text
        tokenized_review = self.tokenizer.tokenize_sentence(review_text)

        # flatten to a vector that equals to the vocabulary size
        word_frequency_vector = np.zeros(len(self.word_to_index_map))
        for w in tokenized_review:
            if w in self.word_to_index_map.keys():
                word_frequency_vector[self.word_to_index_map[w]] += 1
            else:
                word_frequency_vector[-1] += 1

        # normalize the raw counts
        word_frequency_vector = word_frequency_vector / word_frequency_vector.sum()

        return word_frequency_vector

    def fit(self, X, y=None):
        self.set_word_index_mapping()
        return self

    def transform(self, X, y=None):
        """ Get the word frequency vectors for all tokenized reviews """
        word_frequency_matrix = np.zeros((len(X), len(self.word_to_index_map)))
        for i, review in enumerate(X):
            word_frequency_matrix[i, :] = self.get_word_frequency_vector(review)
        return word_frequency_matrix
