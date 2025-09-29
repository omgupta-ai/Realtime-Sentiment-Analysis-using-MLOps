import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    """
    Trains and serializes the sentiment analysis model and vectorizer.
    """
    # --- 1. Load Data ---
    print("Loading cleaned data...")
    df = pd.read_csv('cleaned_sentiment_data.csv')

    # Drop rows with missing values in 'cleaned_text' if any
    df.dropna(subset=['cleaned_text'], inplace=True)

    # Define features (X) and target (y)
    X = df['cleaned_text']
    y = df['sentiment']

    # --- 2. Split Data ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Feature Engineering (TF-IDF) ---
    print("Vectorizing text data with TF-IDF...")
    # TfidfVectorizer converts text into a matrix of numerical features.
    # It gives more weight to words that are important to a document.
    vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- 4. Model Training ---
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # --- 5. Model Evaluation ---
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # --- 6. Serialization (Saving the model) ---
    # Create the 'model' directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Define paths for the saved files
    vectorizer_path = os.path.join('model', 'tfidf_vectorizer.joblib')
    model_path = os.path.join('model', 'logistic_regression.joblib')

    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    print("\nPhase 2 Complete! Model and vectorizer have been saved.")

if __name__ == '__main__':
    train_model()