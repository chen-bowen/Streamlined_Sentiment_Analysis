from sentiment_analysis.data.load_reviews import LoadReviews
import numpy as np


class TrainTestSplit:
    """ split the entire dataset into positive and negative reviews """

    def __init__(self, test_set_pcnt=0.10):
        self.test_set_pcnt = test_set_pcnt
        self.reviews = LoadReviews().reviews

    def get_split_data(self):
        """ 
            split the dataset into training and test, return the 
            processed data in X_train, X_test, y_train, y_test,

            X_train and X_test will be lists of reviews,
            y_train and y_test will be np arrays of 1s and 0s
        """
        # find the number of reviews in test set
        num_positive = len(self.reviews["positive"])
        num_negative = len(self.reviews["negative"])
        num_positive_test = int(num_positive * self.test_set_pcnt)
        num_negative_test = int(num_negative * self.test_set_pcnt)

        # X_train and X_test
        X_train = (
            self.reviews["positive"][:(-num_positive_test)]
            + self.reviews["negative"][:(-num_negative_test)]
        )

        X_test = (
            self.reviews["positive"][-num_positive_test:]
            + self.reviews["negative"][-num_negative_test:]
        )

        # y_train and y_test
        y_train = np.array(
            [1] * (num_positive - num_positive_test)
            + [0] * (num_negative - num_negative_test)
        )

        y_test = np.array([1] * num_positive_test + [0] * num_negative_test)

        return X_train, y_train, X_test, y_test
