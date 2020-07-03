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
        custom_transformer,
        custom_model,
        custom_model_params,
        **kwargs,
    ):
        self.transformer_description = transformer_description
        self.model_description = model_description
        self.custom_transformer = custom_transformer
        self.custom_model = custom_model
        self.custom_model_params = custom_model_params
        self.make_model_pipeline()
        super().init(**kwargs)

    def make_model_pipeline(self):
        """ Build a model pipeline using the word frequency vector transformer and lightGBM classifier"""
        self.pipeline = Pipeline(
            steps=[
                (self.transformer_description, self.custom_transformer),
                (self.model_description, self.custom_model(**self.custom_model_params)),
            ]
        )

    def train(self, X_train, y_train):
        """ Train the model using the pipeline constructed """
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """ Predict with the pipeline created and return the predictions"""
        y_pred = self.pipeline.predict(X_test)
        return y_pred
