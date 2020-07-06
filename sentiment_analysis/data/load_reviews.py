from bs4 import BeautifulSoup
from collections import defaultdict
import os
import numpy as np
from itertools import chain
import json


class LoadReviews:
    """ Utility class to load reviews """

    def __init__(
        self, cached_path=os.path.join(os.path.dirname(__file__), "cache/reviews.json")
    ):
        self._init_file_dir = os.path.dirname(__file__)
        self.categories = ["electronics", "dvd", "kitchen_&_housewares", "books"]
        self.cached_path = cached_path
        self.load_reviews()

    def load_reviews(self):
        """ Load all reviews from the data folder """

        self.reviews = defaultdict(dict)
        np.random.seed(7)
        # populate reviews dict
        for review_type in ["positive", "negative"]:
            for cat in self.categories:
                file_path = os.path.join(
                    self._init_file_dir, "reviews/{}/{}.review".format(cat, review_type)
                )
                reviews_raw = BeautifulSoup(
                    open(file_path).read(), features="html.parser"
                )
                self.reviews[review_type][cat] = [
                    review.text for review in reviews_raw.find_all("review_text")
                ]

                # merge all categories into one
            self.reviews[review_type] = list(
                chain(*list(self.reviews[review_type].values()))
            )
            np.random.shuffle(self.reviews[review_type])

        # save tokenized reviews to cache to speedup build process
        with open(self.cached_path, "w") as fp:
            json.dump(self.reviews, fp)
