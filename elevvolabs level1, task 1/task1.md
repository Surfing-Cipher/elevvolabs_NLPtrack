# Sentiment Analysis Project

## Project Overview

This project builds a binary sentiment classification model that categorizes text data as positive or negative. The focus is on leveraging natural language processing (NLP) techniques and machine learning to preprocess, vectorize, and classify text effectively.

It demonstrates the end-to-end process of an NLP pipeline, from raw data cleaning to model training and evaluation, with visual insights into the dataset.

## Covered Topics

- **Text Preprocessing**: Cleaning and normalizing raw text by lowercasing, removing punctuation, stopwords, and performing lemmatization.
- **Feature Engineering**: Converting text into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency).
- **Binary Classification**: Training and evaluating two different classification models.
- **Model Comparison**: Comparing Logistic Regression and Naive Bayes models for accuracy and performance.
- **Data Visualization**: Generating visualizations such as word clouds to analyze the most frequent words in each class.

## Tools and Libraries

- **Python** – Core programming language.
- **Pandas** – Data manipulation and analysis.
- **NLTK / spaCy** – NLP libraries for tokenization, stopword removal, and lemmatization.
- **Scikit-learn** – Machine learning for feature extraction, training, and evaluation.
- **Matplotlib & WordCloud** – Data visualization.

## Dataset

The project uses a subset of the 20 Newsgroups dataset provided by scikit-learn. Two categories (`comp.graphics` and `rec.autos`) are selected to create a binary classification problem.

## Methodology

### 1. Data Loading and Preprocessing

- Load data into a Pandas DataFrame.
- Convert all text to lowercase.
- Remove punctuation and common English stopwords.
- Apply lemmatization to reduce words to their base forms.

### 2. Feature Engineering

Transform the preprocessed text into numerical vectors using `TfidfVectorizer`, assigning importance scores to words based on their frequency relative to the entire corpus.

### 3. Model Training and Evaluation

Split the data into training and test sets. Train and evaluate two models:

- **Logistic Regression** – Linear model for binary classification.
- **Multinomial Naive Bayes** – Probabilistic classifier optimized for text.

Performance is measured using accuracy and a detailed classification report.

### 4. Visualization

Generate word clouds for each class to highlight the most common words and provide intuitive insights into the dataset.
