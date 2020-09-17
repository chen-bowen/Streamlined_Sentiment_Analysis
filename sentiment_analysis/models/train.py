from sentiment_analysis.models.model import StreamlinedModel
from sentiment_analysis.features.tf_idf import TermFrequency_InvDocFrequency
from sentiment_analysis.utils.data_management import load_dataset, save_pipeline
from sentiment_analysis.config import config
from sentiment_analysis import __version__ as _version
import lightgbm as lgb

import logging


_logger = logging.getLogger(__name__)

def train_model_lgbm():
    """function to train the model """

    lightgbm = StreamlinedModel(
        transformer_description="TF-IDF",
        transformer=TermFrequency_InvDocFrequency,
        model_description="LightGBM model",
        model=lgb.LGBMClassifier,
        model_params={
            "application": "binary",
            "objective": "binary",
            "metric": "auc",
            "is_unbalance": "false",
            "boosting": "gbdt",
            "num_leaves": 31,
            "feature_fraction": 0.06,
            "bagging_fraction": 0.67,
            "bagging_freq": 1,
            "learning_rate": 0.05,
            "verbose_eval": 0,
            "n_estimators": 2000,
            "n_jobs": 6,
        },
    )
    lightgbm.train(X_train, y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=lightgbm.pipeline)

if __name__ == "__main__":
    train_model_lgbm()

