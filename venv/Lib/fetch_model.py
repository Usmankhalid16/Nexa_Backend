import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# Load email data from CSV file
emails = pd.read_csv('ea.csv')

print("Columns in dataset:", emails.columns)

# Define a list of keywords for meeting-related emails
meeting_keywords = ['meeting', 'schedule', 'call', 'conference', 'hang out', 'hangout', 'meet', 'meetup', 'meet up',
                    'meet-up', 'appointment']

# Label each email as meeting or non-meeting based on the keywords in the subject line
emails['label'] = emails['Subject'].apply(
    lambda x: 'non-meeting' if pd.isna(x) else 'meeting' if any(keyword in x.lower() for keyword in meeting_keywords) else 'non-meeting')

# Print the number of meeting and non-meeting emails
print(emails['label'].value_counts())

# Plot a bar chart of the number of meeting and non-meeting emails
emails['label'].value_counts().plot(kind='bar')
plt.show()

# Define keywords for identifying meeting emails
meeting_keywords = ['meeting', 'schedule', 'call', 'conference', 'hang out', 'hangout', 'meet', 'meetup', 'meet up',
                    'meet-up', 'appointment']


def preprocess_text(text):
    if pd.isna(text):
        return ""
    else:
        return text


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails['Subject'], emails['label'], test_size=0.2, random_state=42)

# Preprocess the training and testing data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Create a count vectorizer to convert email subjects into feature vectors
vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_features=5000)

# Fit the vectorizer to the training data and transform the training and testing data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression classifier on the training data
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Predict labels for the testing data
y_pred = clf.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model to disk
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Save the vectorizer to disk
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Print the evaluation metric
print('Accuracy:', accuracy)
