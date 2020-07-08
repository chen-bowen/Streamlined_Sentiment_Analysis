import pandas as pd
import numpy as np
from sentiment_analysis.utils.word_tokenizer import WordTokenizer
from sentiment_analysis.data.load_reviews import LoadReviews
from collections import defaultdict, Counter
import os
from itertools import chain
import json


class ReviewProcessor:
    """ generate processed reviews and word index mapping """

    def __init__(self):
        self._init_file_dir = os.path.dirname(__file__)
        self.tokenizer = WordTokenizer()
        self.build()

    def __tokenize_all_reviews(self, cached_path):
        """" Tokenize all reviews, preprocess the reviews using custom tokenizer """
        self.reviews_tokenized = defaultdict(dict)
        for review_type in ["positive", "negative"]:
            self.reviews_tokenized[review_type] = [
                self.tokenizer.tokenize_sentence(i) for i in self.reviews[review_type]
            ]

        # save tokenized reviews to cache to speedup build process
        with open(cached_path, "w") as fp:
            json.dump(self.reviews_tokenized, fp)

    def __build_word_index_mapping(self, cached_path):
        """ Build word index mapping from all vocabularies """
        all_unique_words = list(
            sorted(
                set(
                    list(chain(*self.reviews_tokenized["positive"]))
                    + list(chain(*self.reviews_tokenized["negative"]))
                )
            )
        )

        self.word_to_index_map = {word: i for i, word in enumerate(all_unique_words)}

        # add a special token to represent unknown word
        self.word_to_index_map["unknown_word"] = len(self.word_to_index_map) + 1
        self.vocab_size = len(self.word_to_index_map)

        # save tokenized reviews to cache to speedup build process
        with open(cached_path, "w") as fp:
            json.dump(self.word_to_index_map, fp)

    def build(self):
        """ Tokenize and build the word to index mapping, word to vector mapping"""
        # load reviews
        cached_path_reviews = os.path.join(self._init_file_dir, "cache/reviews.json")

        # use cached file if exists
        if os.path.exists(cached_path_reviews):
            with open(cached_path_reviews, "r") as fp:
                self.reviews = json.load(fp)

        else:
            print("Loading reviews ...")
            self.reviews = LoadReviews(cached_path_reviews).reviews
            print("Completed")
            print("-----------------")

        # tokenize reviews
        cached_path_tokenized = os.path.join(
            self._init_file_dir, "cache/reviews_tokenized.json"
        )

        # use cached file if exists
        if os.path.exists(cached_path_tokenized):
            with open(cached_path_tokenized, "r") as fp:
                self.reviews_tokenized = json.load(fp)
        else:
            print("Tokenizing reviews ...")
            self.__tokenize_all_reviews(cached_path_tokenized)
            print("Completed")
            print("-----------------")

        # build word to index mapping, which is later used to map the word frequency column index to words
        cached_path_word_index_mapping = os.path.join(
            self._init_file_dir, "cache/word_index_mapping.json"
        )
        # use cached file if exists
        if os.path.exists(cached_path_word_index_mapping):
            with open(cached_path_word_index_mapping, "r") as fp:
                self.word_to_index_map = json.load(fp)
        else:
            print("Building word to index map ...")
            self.__build_word_index_mapping(cached_path_word_index_mapping)
            print("Completed")
            print("-----------------")

