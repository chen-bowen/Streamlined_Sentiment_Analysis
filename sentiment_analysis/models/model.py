import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.features.word_frequencies import WordFrequencyVectorizer


class LightGBMStreamlined:
    def __init__(self):
        self.make_model_pipeline()

    def make_model_pipeline(self):
        """ Bulid a model pipeline using the word frequency vector transformer and lightGBM classifier"""
        self.pipeline = Pipeline(
            steps=[
                ("word_frequency_vectors", WordFrequencyVectorizer()),
                ("lightGBM_model", lgb.LGBMClassifier()),
            ]
        )

    def train(self, X_train, y_train):
        """ Train the model using the pipeline constructed """
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """ Predict with the pipeline created and return the predictions"""
        y_pred = self.pipeline.predict(X_test)
        return y_pred
