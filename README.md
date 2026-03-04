# Automated-review-rating-system

## Project Goal
Build a system that predicts product review ratings (1–5 stars) from text.

## Dataset
100,000 product reviews from HuggingFace dataset.

## Preprocessing
- Lowercasing
- Removing URLs
- Removing punctuation
- Stopword removal
- Lemmatization

## Models
Model A – trained on balanced dataset  
Model B – trained on imbalanced dataset

## Algorithm
LSTM (Deep Learning using TensorFlow/Keras)

## Evaluation
Models evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Web Interface
A Flask web application allows users to enter a review and receive predicted rating.
