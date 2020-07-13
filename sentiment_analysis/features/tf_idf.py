from sklearn.base import BaseEstimator
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.utils.word_tokenizer import WordTokenizer
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


class TermFrequency_InvDocFrequency(BaseEstimator):
    """ Custom TF-IDF transformer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = WordTokenizer()

    def set_word_index_mapping(self, reviews_text):
        """ Initialize the review processor with the inputted reviews text """
        processed_review = ReviewProcessor(reviews_text)
        self.word_to_index_map = processed_review.word_to_index_map
        self.all_unique_words = processed_review.all_unique_words

    def get_tokenized_reviews(self, reviews_text):
        """ Set to all tokenized documents attr with input reviews text """
        tokenized_reviews = [self.tokenizer.tokenize_sentence(s) for s in reviews_text]
        return tokenized_reviews

    def get_word_frequency_vector(self, tokenized_review):
        """ Get the word frequency vector for one tokenized review"""
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

    def get_document_frequency(self, tokenized_reviews):
        """ obtain the document frequency of each word """
        # calculate num of existences for each unique words
        documents_col = pd.DataFrame(
            {"docs": [" ".join(review) for review in tokenized_reviews]}
        )
        document_frequency = defaultdict(int)
        for word in self.all_unique_words:
            document_frequency[word] = documents_col.docs.str.contains(word).sum()

        return document_frequency

    def fit(self, X, y=None):
        """ save inverse document frequency """
        # set word index mapping and get all unique words
        self.set_word_index_mapping(X)

        # idf = log(N/(1 + df))
        self.num_documents = len(X)
        document_frequency = self.get_document_frequency(X)

        self.idfs = {
            word: np.log(self.num_documents / (doc_freq + 1))
            for word, doc_freq in document_frequency.items()
        }

        return self

    def transform(self, X):
        """ Transform the input X"""
        # get tokenized document
        tokenized_reviews = self.get_tokenized_reviews(X)

        # find term frequencies
        word_frequency_matrix = np.zeros((len(X), len(self.word_to_index_map)))
        for i, review in enumerate(tokenized_reviews):
            word_frequency_matrix[i, :] = self.get_word_frequency_vector(review)

        # convert document frequencies into matrix
        idf_vec = np.zeros(len(self.word_to_index_map))
        for word in self.idfs.keys():
            idf_vec[self.word_to_index_map[word]] = self.idfs[word]

        # tile the vector into the same shape as the tf matrix
        idf_matrix = np.tile(idf_vec, (len(X), 1))

        # find tf idf
        tf_idf = word_frequency_matrix * idf_matrix
        return tf_idf
