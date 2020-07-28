# Streamlined NLP Pipeline Modeling on Sentiment Analysis

The classic sentiment analysis project, now with a slight different twist in the level of challenge. 

In this project, we will build a streamlined sentiment analysis model with sklearn pipeline and various machine learning models that we are all familiar with. This streamlined pipeline will have the ability to take in raw text corpus and return the scored sentiment. The specific explanations of implementation of this project are summarized in the two-part blog series on medium below.

[Understanding Text Vectorizations I: How Having a Bag of Words Already Shows What People Think About Your Product](https://towardsdatascience.com/understanding-text-vectorizations-how-streamlined-models-made-feature-extractions-a-breeze-8b9768bbd96a)

[Understanding Text Vectorizations II: How TF-IDF Gives Your Simple Models the Power to Rival the Advanced Ones](https://towardsdatascience.com/understanding-text-vectorizations-ii-how-tf-idf-gives-your-simple-models-the-power-to-rival-the-79b6c975d7eb)

## Environment Setup

1. Clone the repo locally
2. In your terminal, navigate to the directory where you have just cloned this repo
3. Type `pip install poetry`. This will install the package management system poetry which will help you install all the required dependencies
4. Type `poetry install`
5. Type `pip install jupyter`. Now you should be able to view the notebook that generated the above analysis

## Streamlined Model

The `StreamlinedModel` object allow us to build a transformer-model pipeline structure. This pipeline has the flexibility to use any transformer/model combinantion. For example, we can use the sklearn built-in CountVectorizer and logistic regression model to build a pipeline as the following

```python
logistic = StreamlinedModel(
    transformer_description="Bag of words",
    transformer=CountVectorizer,
    model_description="logisitc regression model",
    model=LogisticRegression,
)
```

## Customized Transformers

We will build our own bag of word and TF-IDF transformer, which could be used to input transformer argument to the `StreamlinedModel`. 
