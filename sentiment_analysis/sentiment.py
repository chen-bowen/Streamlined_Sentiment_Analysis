# This code is for my NLP Udemy class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python
# It is written in such a way that tells a story.
# i.e. So you can follow a thought process of starting from a
# simple idea, hitting an obstacle, overcoming it, etc.
# i.e. It is not optimized for anything.

# Author: http://lazyprogrammer.me
from __future__ import print_function, division

# from future.utils import iteritems
from builtins import range
from itertools import chain

# Note: you may need to update your version of future
# sudo pip install -U future


import nltk
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from bs4 import BeautifulSoup
from data.load_reviews import LoadReviews


# wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = stopwords.words("english")

# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
# positive_reviews = []
# negative_reviews = []
# for i in ["books", "dvd", "electronics", "kitchen_&_housewares"]:
#     positive_review = BeautifulSoup(
#         open("sentiment_analysis/data/reviews/{}/positive.review".format(i)).read(),
#         features="html.parser",
#     )
#     positive_reviews.append(positive_review.find_all("review_text"))

#     negative_review = BeautifulSoup(
#         open("sentiment_analysis/data/reviews/{}/negative.review".format(i)).read(),
#         features="html.parser",
#     )
#     negative_reviews.append(negative_review.find_all("review_text"))

# positive_reviews = list(chain.from_iterable(positive_reviews))
# negative_reviews = list(chain.from_iterable(negative_reviews))

# first let's just try to tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
#
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.
# so let's create a function that does all this pre-processing for us

reviews = LoadReviews().reviews
positive_reviews = reviews["positive"]
negative_reviews = reviews["negative"]


def my_tokenizer(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [
        t for t in tokens if len(t) > 2
    ]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review)
    tokens = my_tokenizer(review)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review)
    tokens = my_tokenizer(review)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))

# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # normalize it before setting label
    x[-1] = label
    return x


N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
# import pdb

# pdb.set_trace()


orig_reviews, data = shuffle(orig_reviews, data)

X = data[:, :-1]
Y = data[:, -1]

# last 100 rows will be test
Xtrain = X[:-500,]
Ytrain = Y[:-500,]
Xtest = X[-500:,]
Ytest = Y[-500:,]


parameters = {
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
}
model = lgb.LGBMClassifier(**parameters)
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))


# let's look at the weights for each word
# try it with different threshold values!
# threshold = 0.5
# for word, index in word_index_map.items():
#     weight = model.coef_[0][index]
#     if weight > threshold or weight < -threshold:
#         print(word, weight)


# check misclassified examples
preds = model.predict(X)
P = model.predict_proba(X)[:, 1]  # p(y = 1 | x)

# since there are many, just print the "most" wrong samples
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print(
    "Most wrong positive review (prob = %s, pred = %s):"
    % (minP_whenYis1, wrong_positive_prediction)
)
print(wrong_positive_review)
print(
    "Most wrong negative review (prob = %s, pred = %s):"
    % (maxP_whenYis0, wrong_negative_prediction)
)
print(wrong_negative_review)
