---
title: NYC Taxi Fare Predictor
emoji: 🚕
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: Predict NYC taxi fares using ANN (MLPRegressor)
---

# 🚕 NYC Taxi Fare Predictor

Predict NYC yellow taxi fares using an **Artificial Neural Network (MLPRegressor)** trained on the NYC TLC Trip Records dataset.

## Live Demo
**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/fawazasif/nyc-taxi-fare-predictor)**

## Dataset
**[NYC Taxi Fare Prediction - Kaggle](https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers)**

## Features
- Real-time fare prediction based on trip parameters
- 22 engineered features including time, distance, and location
- Interactive Gradio interface with dark teal theme
- Comprehensive model performance metrics

## Model
- **Algorithm:** MLPRegressor (scikit-learn)
- **Architecture:** 128 → 64 → 32 neurons with ReLU activation
- **Optimizer:** Adam with adaptive learning rate and early stopping

## Student
- **Name:** M FAWAZ ASIF
- **Reg No:** B23F0115CS070
- **University:** Pak Austria Fachhochschule
- **Course:** Machine Learning - Assignment 4
