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

    def __init__(
        self,
        categories=["electronics", "dvd", "kitchen_&_housewares", "books"],
        option="all",
    ):
        self.categories = categories
        self.option = option
        self._init_file_dir = os.path.dirname(__file__)
        self.build()

    def __tokenize_all_reviews(self):
        """" Tokenize all reviews, preprocess the reviews using custom tokenizer """
        self.reviews_tokenized = defaultdict(dict)
        tokenizer = WordTokenizer()
        for review_type in ["positive", "negative"]:
            if self.option != "all":
                for cat in self.categories:
                    self.reviews_tokenized[review_type][cat] = [
                        tokenizer.tokenize_sentence(i)
                        for i in self.reviews[review_type][cat]
                    ]
            else:
                self.reviews_tokenized[review_type] = [
                    tokenizer.tokenize_sentence(i) for i in self.reviews[review_type]
                ]

        # save tokenized reviews to cache to speedup build process
        with open(self.cached_path, "w") as fp:
            json.dump(self.reviews_tokenized, fp)

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
            self.__tokenize_reviews(cached_path_tokenized)
            print("Completed")
            print("-----------------")

        # build word to index mapping, which is later used to map the word frequency column index to words
        all_unique_words = list(
            set(
                list(chain(*self.reviews_tokenized["positive"]))
                + list(chain(*self.reviews_tokenized["negative"]))
            )
        )

        self.word_to_index_map = {word: i for i, word in enumerate(all_unique_words)}
        self.vocab_size = len(self.word_to_index_map)

        # add a special token to represent unknown word
        self.word_to_index_map["unknown_word"] = self.vocab_size + 1
