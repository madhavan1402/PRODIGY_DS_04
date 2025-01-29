import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Load the datasets
twitter_train = pd.read_csv('twitter_training.csv', header=None)
twitter_validation = pd.read_csv('twitter_validation.csv', header=None)

# Print first few rows to check structure
print("Training Data Sample:")
print(twitter_train.head())

# Assign column names dynamically based on the dataset
if twitter_train.shape[1] == 4:
    twitter_train.columns = ['Index', 'Sentiment', 'Label', 'Text']
    twitter_train.drop(columns=['Index', 'Label'], inplace=True)  # Remove unnecessary columns
else:
    twitter_train.columns = ['Index', 'Sentiment', 'Text']
    twitter_train.drop(columns=['Index'], inplace=True)

# Repeat for validation data
if twitter_validation.shape[1] == 4:
    twitter_validation.columns = ['Index', 'Sentiment', 'Label', 'Text']
    twitter_validation.drop(columns=['Index', 'Label'], inplace=True)
else:
    twitter_validation.columns = ['Index', 'Sentiment', 'Text']
    twitter_validation.drop(columns=['Index'], inplace=True)

# Display dataset info
print("Training Data Info:")
print(twitter_train.info())
print("\nTraining Data Sample:")
print(twitter_train.head())

# Check for missing values
twitter_train.dropna(subset=['Sentiment', 'Text'], inplace=True)
twitter_validation.dropna(subset=['Sentiment', 'Text'], inplace=True)

# Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=twitter_train, hue='Sentiment', palette='coolwarm', order=twitter_train['Sentiment'].value_counts().index, legend=False)
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('sentiment_distribution.png')
plt.show()

# Initialize NLTK Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    """Get sentiment polarity score"""
    return sia.polarity_scores(text)['compound']

# Apply sentiment analysis to text
twitter_train['Sentiment_Score'] = twitter_train['Text'].apply(get_sentiment_score)

# Sentiment Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(twitter_train['Sentiment_Score'], bins=30, kde=True, color='purple')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('sentiment_score_distribution.png')
plt.show()

# Generate WordCloud for positive and negative sentiments
positive_text = ' '.join(twitter_train[twitter_train['Sentiment'] == 'Positive']['Text'].dropna())
negative_text = ' '.join(twitter_train[twitter_train['Sentiment'] == 'Negative']['Text'].dropna())

if positive_text.strip():
    plt.figure(figsize=(12, 6))
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Sentiment Word Cloud')
    plt.savefig('positive_wordcloud.png')
    plt.show()
else:
    print("No positive words available for WordCloud.")

if negative_text.strip():
    plt.figure(figsize=(12, 6))
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Sentiment Word Cloud')
    plt.savefig('negative_wordcloud.png')
    plt.show()
else:
    print("No negative words available for WordCloud.")

print("\nSentiment Analysis and Visualization Completed.")
