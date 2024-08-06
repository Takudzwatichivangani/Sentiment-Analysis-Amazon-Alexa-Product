import streamlit as st
import pandas as pd
import pickle
import time
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')  # Download NLTK stopwords dataset

st.title('Sentiment Analysis (Amazon Alexa)')

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        stemmer = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        return review
    else:
        return ''

# Load model
model = pickle.load(open('Model_rf.pkl', 'rb'))

# Load the Count Vectorizer
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

# Function to predict sentiment
def predict_sentiment(review):
    # Preprocess the input
    processed_review = preprocess_text(review)
    
    if processed_review != '':
        # Transform the preprocessed text into numerical features using CountVectorizer
        review_features = cv.transform([processed_review])
        
        # Predict using the model
        prediction = model.predict(review_features)
        
        # Convert prediction to 1 for positive and 0 for negative
        prediction_label = "Positive" if prediction[0] == 1 else "Negative"
        
        return prediction_label
    else:
        return 'N/A'

# Text input prediction
Review = st.text_input('Enter your review')
submit = st.button('Predict')

if submit:
    start = time.time()
    
    # Perform prediction
    prediction_label = predict_sentiment(Review)
    
    # Display prediction
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    st.write("Predicted sentiment is: ", prediction_label)

# File upload prediction
uploaded_file = st.file_uploader("Upload TSV file", type=['tsv'])

if uploaded_file is not None:
    # Read the uploaded TSV file
    df = pd.read_csv(uploaded_file, sep='\t')
    
    # Preprocess the reviews and predict sentiments
    df['Predicted Sentiment'] = df['verified_reviews'].apply(predict_sentiment)
    
    # Display prediction results
    st.write("Prediction results:")
    st.write(df)
    
    # Display a graph showing the distribution of positive and negative sentiments
    sentiment_distribution = df['Predicted Sentiment'].value_counts()
    st.bar_chart(sentiment_distribution)