import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.features.word_frequencies import WordFrequencyVectorizer


class StreamlinedModel:
    """ 
    Streamlined model pipeline
    """

    def __init__(
        self,
        transformer_description,
        model_description,
        transformer,
        model,
        model_params={},
        **kwargs,
    ):
        self.transformer_description = transformer_description
        self.model_description = model_description
        self.custom_transformer = transformer
        self.custom_model = model
        self.custom_model_params = model_params
        self.make_model_pipeline()
        super().__init__(**kwargs)

    def make_model_pipeline(self):
        """ Build a model pipeline using the transformer and the model"""
        self.pipeline = Pipeline(
            steps=[
                (self.transformer_description, self.custom_transformer()),
                (self.model_description, self.custom_model(**self.custom_model_params)),
            ]
        )

    def train(self, X_train, y_train):
        """ Train the model using the pipeline constructed """
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        """ Predict with the pipeline created and return the predictions"""
        y_pred = self.pipeline.predict(X)
        return y_pred

    def predict_proba(self, X):
        """ Get the prediction probabilities """
        y_prob = self.pipeline.predict_proba(X)
        return y_prob

    def score(self, X, y):
        """ Generate preliminary scores from the pipeline """
        accuracy_score = self.pipeline.score(X, y)
        return accuracy_score
