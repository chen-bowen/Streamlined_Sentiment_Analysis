import numpy as np
import pandas as pd

from sentiment_analysis.utils.data_management import load_pipeline
from sentiment_analysis.config import config
# from sentiment_analysis.processing.validation import validate_inputs
from sentiment_analysis import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_sentiment_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data):
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(validated_data[config.FEATURES])

    output = np.exp(prediction)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results
