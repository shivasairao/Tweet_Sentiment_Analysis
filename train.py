import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_tweet
from vectorization import MeanEmbeddingVectorizer

def main():
    # Setup directory paths based on the current folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print("Reading Target Dataset...")
    # Explicitly point to Tweets.csv in the exact same folder as this script
    csv_path = os.path.join(current_dir, "Tweets.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text', 'airline_sentiment'])
    
    print("Preprocessing text collections...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    df = df[df['cleaned_text'].map(len) > 0] # Filter empty documents
    
    X = df['cleaned_text'].tolist()
    y = df['airline_sentiment'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Optimizing Continuous Bag-of-Words Vector Space...")
    vectorizer = MeanEmbeddingVectorizer(vector_size=200, window=5, min_count=2)
    vectorizer.fit(X_train)
    
    print("Vectorizing Text Features...")
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Downstream Classification Engine...")
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X_train_vec, y_train)
    
    # Validation Analytics
    y_pred = classifier.predict(X_test_vec)
    print("\n================ EVALUATION METRICS ================")
    print(f"Overall Dataset Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    print("Serializing Project Artifacts...")
    vectorizer.save(os.path.join(model_dir, "w2v_model.model"))
    joblib.dump(classifier, os.path.join(model_dir, "sentiment_classifier.pkl"))
    print("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    main()