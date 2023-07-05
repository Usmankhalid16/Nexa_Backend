from __future__ import print_function
import base64
import time
import datefinder
import os.path
import re
import datetime
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pickle
from sklearn.feature_extraction.text import CountVectorizer

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Define meeting-related keywords
meeting_keywords = ['meeting', 'schedule', 'call', 'conference', 'hang out', 'hangout', 'meet', 'meetup', 'meet up',
                    'meet-up', 'appointment']


def classify_email(email_subject):
    # Preprocess the email subject
    processed_subject = preprocess_text(email_subject)

    # Convert the preprocessed subject into a feature vector
    subject_vec = vectorizer.transform([processed_subject])

    # Predict the label using the trained model
    label = model.predict(subject_vec)[0]

    return label


def preprocess_text(text):
    if pd.isna(text):
        return ""
    else:
        return text


def funct():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    emails = []
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)

        # Getting emails
        query = "newer_than:7d "
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q=query).execute()
        messages = results.get('messages', [])

        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            emailSnippet = msg["snippet"]
            msgType = ""
            msgBody = ""
            msgSubject = ""

            for header in msg["payload"]["headers"]:
                if header["name"] == "Subject":
                    msgSubject = header["value"]
                    break

            if "parts" in msg["payload"]:
                msgType = msg["payload"]["parts"][0]["mimeType"]
                if msgType == "multipart/alternative":
                    msgBody = msg["payload"]["parts"][0]["parts"][0]["body"]["data"]
                elif msgType == "text/plain":
                    msgBody = msg["payload"]["parts"][0]["body"]["data"]
            else:
                msgType = msg["payload"]["mimeType"]
                msgBody = msg["payload"]["body"]["data"]

            msgBodyDecoded = base64.urlsafe_b64decode(msgBody)

            # Check for keywords in subject
            if classify_email(msgSubject) == 'meeting' or "meet.google.com" in str(msgBodyDecoded)or any(keyword in msgSubject.lower() for keyword in ["meeting", "schedule", "call", "conference"]) :
                # Process the email body as per your requirements
                if "meet.google.com" in str(msgBodyDecoded):
                    # Process Google Meet emails
                    res = msgBodyDecoded.decode("utf-8")
                    split = res.split("\r\n\r\n")
                    if "Unknown sender" in split[0]:
                        data = {
                            "title": split[1].split("\r\n")[0],
                            "date": split[1].split("\r\n")[1].split("⋅")[0],
                            "start_time": split[1].split("\r\n")[1].split("⋅")[1].split("–")[0].strip(),
                            "end_time": split[1].split("\r\n")[1].split("⋅")[1].split("–")[1].strip()
                        }
                        emails.append(data)
                        print(data)
                    else:
                        data = {
                            "title": split[1].split("\r\n")[0],
                            "date": split[1].split("\r\n")[1].split("⋅")[0],
                            "start_time": split[1].split("\r\n")[1].split("⋅")[1].split("–")[0].strip(),
                            "end_time": split[1].split("\r\n")[1].split("⋅")[1].split("–")[1].strip()
                        }
                        emails.append(data)
                        print(data)
                else:
                    # Process emails with meeting-related keywords in the subject
                    title = re.search(r"Subject: (.*)", str(msgBodyDecoded))
                    matches = list(datefinder.find_dates(msgBodyDecoded.decode()))
                    if len(matches) > 0:
                        dt = matches[0]
                        date = dt.strftime("%m/%d/%Y")
                        time = dt.strftime("%I:%M %p")
                        timezone = datetime.datetime.strftime(dt, "%Z")
                        description = msgBodyDecoded.decode("utf-8").strip()
                        data = {
                            "title": msgSubject,
                            "date": date,
                            "start_time": time,
                            "description": description
                        }
                        emails.append(data)
                        print(data)
                    else:
                        print("No dates or times found.")

        return emails

    except HttpError as error:
        print(f'An error occurred: {error}')

# if __name__ == '__main__':
#     funct()