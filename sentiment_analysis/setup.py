from setuptools import setup

setup(
    name="sentiment_analysis",
    version="1.0",
    description="Application of NLP and Genetic Algorithm in cipher decryption",
    author="Bowen Chen",
    packages=["sentiment_analysis"],  # same as name
    install_requires=[
        "pandas",
        "numpy",
        "ipykernel",
        "lightgbm",
        "shap",
        "sklearn",
        "beautifulsoup4",
    ],  # external packages as dependencies
)
