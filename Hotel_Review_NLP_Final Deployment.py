import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit title
st.title('Hotel Reviews Analysis')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data into a DataFrame
        data_Hotel_review = pd.read_csv(uploaded_file)
        
        # Preprocess text data
        data_Hotel_review['Processed_Review'] = data_Hotel_review['Review'].apply(preprocess_text)

        st.header('Data Preview')
        st.write(data_Hotel_review.head())  # Show the first few rows of the dataframe

        # Parking Feedback Analysis
        st.header('Parking Feedback')
        parking_feedback = data_Hotel_review[data_Hotel_review['Review'].str.contains('parking', case=False)]

        # Count positive and negative parking feedbacks
        positive_count = parking_feedback[parking_feedback['Feedback'] == 'Pos'].shape[0]
        negative_count = parking_feedback[parking_feedback['Feedback'] == 'Neg'].shape[0]

        fig, ax = plt.subplots()
        bars = ax.bar(['Positive', 'Negative'], [positive_count, negative_count], color=['skyblue', 'red'])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=12)
        ax.set_xlabel('Feedback')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Parking Feedback')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        # Cost Perception Analysis
        st.header('Cost Perception')
        expensive_reviews = data_Hotel_review[data_Hotel_review['Review'].str.contains('expensive|pricey', case=False)]
        affordable_reviews = data_Hotel_review[data_Hotel_review['Review'].str.contains('affordable|cheap', case=False)]

        expensive_count = expensive_reviews.shape[0]
        affordable_count = affordable_reviews.shape[0]

        fig, ax = plt.subplots()
        bars = ax.bar(['Expensive', 'Affordable'], [expensive_count, affordable_count], color=['green', 'yellow'])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=12)
        ax.set_title('Perceived Cost of the Hotel')
        ax.set_ylabel('Number of Reviews')
        ax.grid(axis='y')

        st.pyplot(fig)

        # Booking Sites Analysis
        st.header('Top Booking Sites')
        booking_sites = data_Hotel_review['Review'].str.extract(r'(\w+\.com)')[0].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(y=booking_sites.head(5).index, x=booking_sites.head(5), palette='viridis', ax=ax)
        ax.set_title('Top 5 Websites for Booking a Hotel')
        ax.set_xlabel('Number of Customers Using the Website')
        ax.set_ylabel('Website Names')

        for index, value in enumerate(booking_sites.head(5)):
            ax.text(value, index, str(value), va='center', ha='left', fontsize=12)

        st.pyplot(fig)

        # Trip Type Analysis
        st.header('Trip Type Analysis')
        business_trip_count = data_Hotel_review[data_Hotel_review['Review'].str.contains('business trip', case=False)].shape[0]
        family_trip_count = data_Hotel_review[data_Hotel_review['Review'].str.contains('family trip', case=False)].shape[0]

        fig, ax = plt.subplots()
        bars = ax.bar(['Business Trip', 'Family Trip'], [business_trip_count, family_trip_count], color=['purple', 'pink'])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), va='bottom', ha='center', fontsize=12)
        ax.set_title('Number of People by Trip Type')
        ax.set_ylabel('Count')
        ax.grid(axis='y')

        st.pyplot(fig)

        # Distribution of Ratings as a Pie Chart
        st.header('Distribution of Ratings')
        plt.figure(figsize=(8, 8))
        labels = data_Hotel_review['Feedback'].value_counts().index
        sizes = data_Hotel_review['Feedback'].value_counts().values
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'pink', 'yellow', 'green', 'red'])
        plt.title('Distribution of Ratings')
        plt.axis('equal') 
        st.pyplot(plt)

        # Logistic Regression Model
        st.header('Sentiment Analysis using Logistic Regression')
        
        # Convert text to numerical data
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data_Hotel_review['Processed_Review'])
        y = data_Hotel_review['Feedback'].apply(lambda x: 1 if x == 'Pos' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)

        # Evaluation
        y_pred_test_leg_reg = lr_model.predict(X_test)
        y_pred_train_leg_reg = lr_model.predict(X_train)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred_test_leg_reg))

        # Calculate accuracy
        Test_accuracy = accuracy_score(y_test, y_pred_test_leg_reg)
        Train_accuracy = accuracy_score(y_train, y_pred_train_leg_reg)
        st.write(f"Test Logistic Regression Accuracy: {Test_accuracy * 100:.2f}%")
        st.write(f"Train Logistic Regression Accuracy: {Train_accuracy * 100:.2f}%")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        st.write("Confusion Matrix for Logistic Regression")
        st.write(confusion_matrix(y_test, y_pred_test_leg_reg))

        # ROC Curve
        st.subheader("ROC Curve")
        y_prob = lr_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(plt)

        # Cross-Validation Scores
        st.subheader("Cross-Validation Scores")
        scores = cross_val_score(lr_model, X, y, cv=5)
        st.write(f'Cross-Validation Scores: {scores}')
        st.write(f'Average Cross-Validation Score: {scores.mean():.2f}')

        # Add a text input box for user comments
        st.header('Comment Sentiment Analysis')
        user_comment = st.text_input("Type your review here:")

        if user_comment:
            processed_comment = preprocess_text(user_comment)
            vectorized_comment = vectorizer.transform([processed_comment])
            prediction = lr_model.predict(vectorized_comment)[0]

            if prediction == 1:
                st.success("This comment is predicted to be **Positive**.")
            else:
                st.warning("This comment is predicted to be **Negative**.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info('Please upload a CSV file to start the analysis.')
