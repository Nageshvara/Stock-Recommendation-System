import os
from flask import Flask, request, redirect, url_for, send_from_directory,render_template,jsonify
import json
from collections import Counter
from urllib.parse import quote
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime as dt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = pd.read_csv('mini_pro_data.csv')
data['Open'] = data['Open'].replace({'\$': ''}, regex=True).astype(float)
data['High'] = data['High'].replace({'\$': ''}, regex=True).astype(float)
data['Low'] = data['Low'].replace({'\$': ''}, regex=True).astype(float)
data['Close/Last'] = data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)


X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close/Last']

model = LinearRegression()
model.fit(X, y)

X_tomorrow = data[['Close/Last']]
y_tomorrow = data[['Open', 'High', 'Low', 'Volume']]

model_tomorrow = LinearRegression()
model_tomorrow.fit(X_tomorrow, y_tomorrow)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('login.html')


@app.route('/searchhome', methods=['GET', 'POST'])
def searchhome():
    return render_template('landing.html')

@app.route('/personal', methods=['GET', 'POST'])
def personal():
    return render_template('index.html')

@app.route('/personalresult', methods=['GET', 'POST'])
def personalresult():
    return render_template('personalresult.html')

@app.route('/newsoption', methods=['GET', 'POST'])
def newsoption():
    return render_template('newsoption.html')

nltk.download('vader_lexicon')

# Function to convert company name to ticker symbol
def get_ticker_symbol(company_name):
    # Sample mapping from a publicly available source
    # You may need to use a more comprehensive source or API for accurate mappings
    company_tickers = {
        'Apple': 'AAPL',
        'Google': 'GOOGL',
        'Microsoft': 'MSFT',
        'Amazon': 'AMZN',
        'Facebook': 'FB',
        'Tesla': 'TSLA',
        'IBM': 'IBM',
        'Mahindra': 'MAHMF',  # Replace 'MAHMF' with the actual ticker for Mahindra
        'Tata': 'TATA',  # Replace 'TATA' with the actual ticker for Tata
        'JPMorgan': 'JPM',  # JPMorgan Chase & Co.
        'Johnson & Johnson': 'JNJ',
        'Walmart': 'WMT',
        'Procter & Gamble': 'PG',
        'Coca-Cola': 'KO',
        'Intel': 'INTC',
        'Boeing': 'BA',
        'ExxonMobil': 'XOM',
        'Walt Disney': 'DIS',
        # Add more companies as needed
    }
    return company_tickers.get(company_name, '')

@app.route('/newsresult', methods=['POST'])
def newsresult():
    # Get company name from the form data
    company_name = request.form.get('company_name')

    # Convert company name to ticker symbol
    ticker_symbol = get_ticker_symbol(company_name)

    # Check if a valid ticker symbol is obtained
    if not ticker_symbol:
        return render_template('newsresult.html', error_message="Invalid company name. Please try again.")

    # Generate the URL with the ticker symbol
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker_symbol}&from=2023-08-15&to=2023-08-20&token=claspgpr01qi1291cqd0claspgpr01qi1291cqdg"

    # Fetch news data from the API
    response = requests.get(url)
    data = response.json()

    # Create a SentimentIntensityAnalyzer object
    sid = SentimentIntensityAnalyzer()

    # Store sentiment scores for all articles
    all_sentiment_scores = []

    # Store the latest news headlines containing the keyword
    keyword = company_name  # Use the company name as the keyword
    headlines_with_keyword = [article['headline'] for article in data if keyword.lower() in article['headline'].lower()]

    # Take the top 5 headlines (or less if there are fewer than 5)
    top_5_headlines = headlines_with_keyword[:5]

    # Iterate through each news article in the response
    for article in data:
        # Perform sentiment analysis on the headline
        sentiment_score = sid.polarity_scores(article['headline'])
        # Append the sentiment score to the list
        all_sentiment_scores.append(sentiment_score['compound'])

    # Calculate the average sentiment score
    average_sentiment_score = sum(all_sentiment_scores) / len(all_sentiment_scores)

    # Render the template with the results
    return render_template('newsresult.html', company_name=company_name, keyword=keyword,
                           top_5_headlines=top_5_headlines, average_sentiment_score=average_sentiment_score)


@app.route('/trendsoption', methods=['GET', 'POST'])
def trendsoption():
    return render_template('trendsoption.html')

@app.route('/todayclose', methods=['GET', 'POST'])
def todayclose():
    return render_template('todayclose.html')

@app.route('/tomorrowvalues', methods=['GET', 'POST'])
def tomorrowvalues():
    return render_template('tomorrowvalues.html')

import requests

# Define your Google Custom Search API key and engine ID
API_KEY = 'AIzaSyAYQfizYToOPxNvV-QYKqC_sqv7viy2fOY'
ENGINE_ID = 'c4df414d9f1e147b0'

@app.route('/history', methods=['POST', 'GET'])
def handle_history():
    global search_results  # Use the global search_results variable

    if request.method == 'POST':
        data = request.get_json()
        if data is not None:
            query_history = data.get('queryHistory', [])
            text_values = query_history.values()
            text_blob = ' '.join(text_values)
            words = text_blob.split()
            word_frequencies = Counter(words)
            exclude_words = ["stock", "price", "today"]
            filtered_frequencies = {word: freq for word, freq in word_frequencies.items() if word not in exclude_words}
            filtered_counter = Counter(filtered_frequencies)
            top_three_words = filtered_counter.most_common(3)
            search_results = []

            # Integrate Google Custom Search API to fetch image URLs
            for word, freq in top_three_words:
                google_search_link = f"https://www.google.com/search?q={quote(word)}+stock"
                image_search_url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={ENGINE_ID}&q={quote(word)}+today+stock+graph&searchType=image"
                response = requests.get(image_search_url)
                if response.status_code == 200:
                    image_data = response.json()
                    if 'items' in image_data:
                        first_image_url = image_data['items'][0]['link']
                        search_results.append({'link': google_search_link, 'word': word, 'image_url': first_image_url})
                    else:
                        search_results.append({'link': google_search_link, 'word': word, 'image_url': None})
                else:
                    search_results.append({'link': google_search_link, 'word': word, 'image_url': None})

    # Handle the GET request here, or return an appropriate response
    return render_template('personalresult.html', search_results=search_results)


@app.route('/predict_today_close', methods=['POST'])
def predict_today_close():
    try:
        open_val = float(request.form['open'])
        high_val = float(request.form['high'])
        low_val = float(request.form['low'])
        volume_val = float(request.form['volume'])

        predicted_close = model.predict([[open_val, high_val, low_val, volume_val]])[0]

        return render_template('closeresult.html', prediction_result=f"Predicted Today's Close Price: {predicted_close:.2f}")
    except Exception as e:
        return f'Error: {str(e)}'

@app.route('/predict_tomorrow_open', methods=['POST'])
def predict_tomorrow_open():
    try:

        close_value = float(request.form['close'])

        tomorrow_open_high_low_volume_prediction = model_tomorrow.predict([[close_value]])
    
        return render_template('openresult.html', close=close_value, 
                           tomorrow_open=tomorrow_open_high_low_volume_prediction[0])

    except Exception as e:
        return f'Error: {str(e)}'


if __name__ == '__main__':
    app.run()