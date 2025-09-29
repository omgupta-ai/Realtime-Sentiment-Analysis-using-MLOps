import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_data():
    """
    Loads, preprocesses, and saves the sentiment data.
    """
    # Download necessary NLTK data (only needs to be done once)
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    # --- 1. Load Data ---
    # Define column names as the CSV does not have a header
    column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(
        'training.1600000.processed.noemoticon.csv',
        encoding='latin-1', # This encoding is common for this dataset
        names=column_names
    )

    # --- 2. Data Cleaning & Preprocessing ---
    print("Preprocessing text data...")

    # For simplicity, let's map sentiment '4' to '1' (positive)
    # '0' remains '0' (negative)
    df['sentiment'] = df['sentiment'].replace(4, 1)

    # Define a function to clean each tweet
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def clean_text(text):
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keeping the text)
        text = re.sub(r'#', '', text)
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize and remove stopwords, and apply stemming
        tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
        return " ".join(tokens)

    # Apply the cleaning function to the 'text' column
    df['cleaned_text'] = df['text'].apply(clean_text)

    # --- 3. Save Processed Data ---
    # We only need the sentiment and the cleaned text for our model
    processed_df = df[['sentiment', 'cleaned_text']]
    
    # Let's use a smaller sample for faster training initially.
    # We can use the full dataset later if needed.
    # Taking 100k positive and 100k negative samples.
    df_neg = processed_df[processed_df['sentiment'] == 0].sample(n=100000, random_state=42)
    df_pos = processed_df[processed_df['sentiment'] == 1].sample(n=100000, random_state=42)
    
    final_df = pd.concat([df_neg, df_pos])

    # Save the final dataframe to a new CSV
    output_path = 'cleaned_sentiment_data.csv'
    print(f"Saving cleaned data to {output_path}...")
    final_df.to_csv(output_path, index=False)

    print("Phase 1 Complete! Data has been cleaned and saved.")
    print("\nCleaned Data Head:")
    print(final_df.head())


if __name__ == '__main__':
    preprocess_data()