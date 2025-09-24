# Enhancing Social Media Integrity: A Machine Learning-Based Rumor Identification System Utilizing CNN for Accurate Real-Time Tweets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)

## Project Overview

In today's digital age, social media platforms are not just channels for social interaction but also battlefields for information integrity. The rapid dissemination of rumors on platforms like Twitter (now X) can lead to widespread misinformation, affecting public opinion, health, and societal stability. This project, titled **Enhancing Social Media Integrity: A Machine Learning-Based Rumor Identification System Utilizing CNN for Accurate Real-Time Tweets**, introduces an automated system to detect and classify tweets as rumors or non-rumors in real-time.

Developed as part of a B.Tech project in Computer Science and Engineering at Vellore Institute of Technology (VIT), this system leverages advanced machine learning techniques, with a focus on Convolutional Neural Networks (CNN) for superior performance in handling unstructured text data. The project explores multiple algorithms—including Logistic Regression, Naive Bayes, Random Forest, SVM, LSTM, and CNN—and identifies CNN as the optimal model due to its high accuracy (up to 96%) and efficiency.

Key highlights from the abstract:
- **Problem Addressed**: Rapid spread of misinformation on social media.
- **Approach**: Preprocessing (noise removal, tokenization, vectorization), feature extraction, model training, and real-time deployment.
- **Deployment**: A user-friendly web interface for instant predictions, scalable via cloud platforms.
- **Impact**: Promotes a safer digital community by curbing rumor propagation.

This repository contains the complete implementation, including training scripts, a Flask-based web app, saved models, and documentation. It serves as a reproducible blueprint for rumor detection systems.

## Features

- **Real-Time Rumor Detection**: Input a tweet via the web interface to get instant classification as "Likely True (Non-Rumor)" or "Likely False (Rumor)".
- **Model Comparison**: Evaluates multiple ML/DL models (LR, NB, RF, SVM, LSTM, CNN) with metrics like accuracy, precision, recall, and F1-score.
- **Data Preprocessing Pipeline**: Handles cleaning (URLs, mentions, punctuation), normalization, stemming, and vectorization using TF-IDF and tokenization.
- **Visualizations**: Confusion matrices, accuracy comparison plots, precision-recall curves, and ROC curves.
- **Scalable Deployment**: Flask app ready for local run or cloud hosting (e.g., Heroku, AWS).
- **Modular Architecture**: Easy to extend with future enhancements like multimodal analysis (text + images).

## System Architecture

![System Architecture](docs/system_architecture.png)

The system follows a modular design:
1. **Data Collection & Preprocessing**: Load dataset, clean text.
2. **Feature Extraction**: TF-IDF for traditional models; tokenization/padding for DL.
3. **Model Training & Evaluation**: Train/compare models, generate metrics.
4. **Prediction & UI**: Load saved model for real-time inference via web app.
5. **Deployment & Maintenance**: Cloud-ready with monitoring for updates.

## Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended: `venv` or Conda)

### Steps
1. Clone the repository:
