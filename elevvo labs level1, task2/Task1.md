# News Category Classification

## Project Overview

This project demonstrates a complete machine learning workflow for **Natural Language Processing (NLP)**.  
The goal is to build a robust model that can accurately classify news articles into categories such as **World, Sports, Business, and Sci/Tech**.

---

## Methodology

### 1. Data Preparation

- Load the **AG News dataset**.
- Combine the `title` and `description` fields into a single text feature for analysis.

### 2. Text Preprocessing

- **Cleaning**: Convert text to lowercase, remove punctuation and numbers.
- **Normalization**: Remove stopwords and apply lemmatization to reduce words to their base form.

### 3. Feature Engineering

- Use **TF-IDF Vectorization** to transform text into numerical features.
- TF-IDF assigns weights to words, emphasizing their importance relative to the entire dataset.

### 4. Model Training

- Train an **XGBoost Classifier** on the vectorized data.
- XGBoost is chosen for its efficiency and strong performance on structured data.

### 5. Model Evaluation

- Evaluate the trained model on a held-out test set.
- Metrics include **Accuracy** and a **Classification Report** for per-class performance.

---

## Dataset

The project uses the **AG News Corpus**, a widely used benchmark dataset for text classification.

- **Training samples**: 120,000+
- **Test samples**: 7,600
- **Categories**:
  1. World
  2. Sports
  3. Business
  4. Sci/Tech

---

## Tools and Libraries

- **Python**: Core programming language.
- **Jupyter Notebook**: Interactive development environment.
- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning utilities for data splitting, feature extraction, and evaluation.
- **NLTK**: Text preprocessing (stopword removal, lemmatization).
- **XGBoost**: Gradient boosting algorithm for classification.
- **Matplotlib**: Visualization of results and insights.

---
