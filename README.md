# Sentiment Analysis on Topical Chat Dataset

## Overview
In this project, we aim to build a sentiment analysis model based on the Topical Chat dataset. The dataset contains over 8000 conversations and over 184000 messages, each labeled with one of the following sentiments: Angry, Curious to Dive Deeper, Disguised, Fearful, Happy, Sad, and Surprised.

## Approach
### Data Preprocessing:
- The dataset was loaded using pandas.
- Removed the null data from the dataset.
- Perform the groupby operation on both "conversation_id" and "sentiment" columns.For each unique combination of "conversation_id" and "sentiment", we aggregate the messages into a single string by joining the values with a delimiter(, ).
- Removed the unnecessary text and stopwords for the text and applied Stemming technique.
- The text data was preprocessed using a TF-IDF vectorizer to convert text into numerical features.

### Handling Class Imbalance:
- As the dataset contains multiple classes and may suffer from class imbalance, we applied the Synthetic Minority Over-sampling Technique (SMOTE) to oversample minority classes.

- ## Train Test Split
- The dataset was split into training and testing sets.

### Model Training:
- We trained a Linear Support Vector Classifier (LinearSVC) model on the preprocessed and oversampled training data on specific hyperparameters.
- The model was trained to predict the sentiment label based on the TF-IDF features.
- The model was saved in .sav (sentiment_model_1.sav).

### Evaluation:
- The trained model was evaluated on the testing set using classification metrics such as accuracy, precision, recall, and F1-score.
- The classification report was generated to assess the performance of the model on each sentiment class.

- ## Prediction
- Load the trained model and performed the prediction on unseen data.

## Files:
- `sentiment_analysis.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `Topical_chat.csv`: Dataset file containing conversations and messages with sentiment labels.
- `sentiment_model_1.sav`: Trained Model.
- `README.md`: This file, explaining the approach and providing an overview of the project.
- `requirements.txt`: Contains all the dependencies that requires for model training and prediction.

## Requirements:
- Python 3.x
- Libraries: pandas, scikit-learn, imbalanced-learn, nltk

## Instructions:
1. Install the required libraries using `pip install -r requirements.txt`.
2. Run the `sentiment_analysis.ipynb` notebook to preprocess the data, train the model, and evaluate its performance.

