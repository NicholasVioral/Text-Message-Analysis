import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
import gc

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the CSV file in chunks
csv_file_path = 'clean_nus_sms.csv'
chunk_size = 1000  # Adjust chunk size as needed

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify the sentiment of a message
def classify_sentiment(message):
    scores = sid.polarity_scores(message)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Initialize an empty DataFrame to store processed data
df_processed = pd.DataFrame()

# Process the data in chunks
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    chunk['Message'] = chunk['Message'].fillna('').astype(str)
    chunk['Sentiment'] = chunk['Message'].apply(classify_sentiment)
    df_processed = pd.concat([df_processed, chunk], ignore_index=True)
    del chunk
    gc.collect()

# Display sentiment distribution
sentiment_counts = df_processed['Sentiment'].value_counts(normalize=True) * 100
print("Sentiment Distribution:")
print(sentiment_counts)

# Visualize the sentiment distribution
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution of Messages')
plt.xlabel('Sentiment')
plt.ylabel('Percentage')
plt.show()

# Perform topic modeling with a smaller sample
sample_size = min(1000, len(df_processed))  # Use a maximum of 1000 samples or the full dataset if smaller
df_sample = df_processed.sample(n=sample_size, random_state=42)

vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limit features to improve performance
X = vectorizer.fit_transform(df_sample['Message'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display the top words in each topic
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}:")
    print([feature_names[i] for i in topic.argsort()[-10:]])

# Create frequency distributions for key words/phrases
word_counts = pd.Series(' '.join(df_processed['Message']).split()).value_counts()[:10]
print("Top 10 Words:")
print(word_counts)

# Visualize the word frequency distribution
word_counts.plot(kind='bar')
plt.title('Top 10 Words in Messages')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Text similarity using TF-IDF and cosine similarity with a smaller sample
tfidf_sample = TfidfVectorizer().fit_transform(df_sample['Message'])
cosine_sim = cosine_similarity(tfidf_sample, tfidf_sample)
print("Cosine Similarity Matrix Sample:")
print(cosine_sim)

# POS tagging using SpaCy with a smaller sample
nlp = spacy.load('en_core_web_sm')
df_sample['POS_Tags'] = df_sample['Message'].apply(lambda x: [(token.text, token.pos_) for token in nlp(x)])
print("Sample POS Tags:")
print(df_sample[['Message', 'POS_Tags']].head())

# Clean up memory
del df_sample, tfidf_sample, cosine_sim, X, lda, vectorizer
gc.collect()
