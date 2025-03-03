import os
import pandas as pd
import boto3
from dotenv import load_dotenv

# Load .env File
load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')

comprehend = boto3.client(
    service_name='comprehend',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def detect_sentiment(text):
    if pd.isnull(text):
        return 'Neutral'
    response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return response['Sentiment']

df = pd.read_csv('./Data/Customer_Reviews.csv')
df['Sentiment'] = df['reviewDescription'].apply(detect_sentiment)
df.to_csv('./Output/Customer_Sentiment_Output.csv', index=False)
print("Sentiment Analysis Completed ✅")
