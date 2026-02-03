# ==========================================
# Twitter Sentiment Analysis - ML Version
# Author: Madhavan N
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# Download NLTK Data
# ==========================================
nltk.download('vader_lexicon')


# ==========================================
# Load Dataset
# ==========================================
twitter_train = pd.read_csv('twitter_training.csv', header=None)

print("Training Sample:")
print(twitter_train.head())


# ==========================================
# Assign Column Names
# ==========================================
if twitter_train.shape[1] == 4:
    twitter_train.columns = ['Index', 'Topic', 'Sentiment', 'Text']
    twitter_train.drop(columns=['Index', 'Topic'], inplace=True)
else:
    twitter_train.columns = ['Index', 'Sentiment', 'Text']
    twitter_train.drop(columns=['Index'], inplace=True)


# ==========================================
# Remove Missing Values
# ==========================================
twitter_train.dropna(inplace=True)
twitter_train.reset_index(drop=True, inplace=True)

print("\nDataset Info:")
print(twitter_train.info())


# ==========================================
# Sentiment Distribution
# ==========================================
plt.figure(figsize=(7, 5))
sns.countplot(x='Sentiment', data=twitter_train)
plt.title("Sentiment Distribution")
plt.xticks(rotation=45)
plt.savefig("sentiment_distribution.png")
plt.show()


# ==========================================
# Text Cleaning Function
# ==========================================
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove mentions & hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==========================================
# Apply Cleaning
# ==========================================
twitter_train['Clean_Text'] = twitter_train['Text'].apply(clean_text)

# Remove very short texts
twitter_train = twitter_train[
    twitter_train['Clean_Text'].str.len() > 3
]

twitter_train.reset_index(drop=True, inplace=True)

print("Samples after cleaning:", len(twitter_train))


# ==========================================
# VADER Sentiment Analysis
# ==========================================
sia = SentimentIntensityAnalyzer()

twitter_train['VADER_Score'] = twitter_train['Text'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)

plt.figure(figsize=(7, 5))
sns.histplot(twitter_train['VADER_Score'], bins=30, kde=True)
plt.title("VADER Score Distribution")
plt.savefig("vader_distribution.png")
plt.show()


# ==========================================
# Encode Labels
# ==========================================
label_map = {
    'Positive': 1,
    'Negative': 0,
    'Neutral': 2
}

twitter_train = twitter_train[
    twitter_train['Sentiment'].isin(label_map.keys())
]

twitter_train['Label'] = twitter_train['Sentiment'].map(label_map)


# ==========================================
# Feature Extraction (TF-IDF)
# ==========================================
X = twitter_train['Clean_Text']
y = twitter_train['Label']

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_vec = tfidf.fit_transform(X)


# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# Train Model
# ==========================================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# ==========================================
# Prediction & Evaluation
# ==========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================
# Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()


# ==========================================
# WordCloud
# ==========================================
positive_text = " ".join(
    twitter_train[twitter_train['Sentiment'] == 'Positive']['Text']
)

negative_text = " ".join(
    twitter_train[twitter_train['Sentiment'] == 'Negative']['Text']
)


if positive_text.strip():

    wc_pos = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(positive_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc_pos)
    plt.axis("off")
    plt.title("Positive WordCloud")
    plt.savefig("positive_wordcloud.png")
    plt.show()


if negative_text.strip():

    wc_neg = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='Reds'
    ).generate(negative_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc_neg)
    plt.axis("off")
    plt.title("Negative WordCloud")
    plt.savefig("negative_wordcloud.png")
    plt.show()


# ==========================================
# Compare VADER vs ML
# ==========================================
print("Average VADER Score:",
      round(twitter_train['VADER_Score'].mean(), 3))

print("ML Accuracy:",
      round(accuracy * 100, 2), "%")


# ==========================================
# Done
# ==========================================
print("\nSentiment Analysis Completed Successfully!")
