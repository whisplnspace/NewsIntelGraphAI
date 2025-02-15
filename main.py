import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import numpy as np
from langdetect import detect
from googletrans import Translator
import plotly.express as px
import yake
import pyttsx3
from collections import Counter
import pandas as pd
import google.generativeai as genai
import os

# **Set Streamlit Page Configuration**
st.set_page_config(page_title="NewsIntelGraph", page_icon="📰", layout="wide")

# **Load API Keys from Streamlit Secrets**
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# **Download NLTK Dependencies**
nltk.download('punkt')


# **Load SpaCy NLP Model (Cached)**
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_lg")


nlp = load_spacy()

# **Initialize Text-to-Speech Engine**
tts_engine = pyttsx3.init()


def text_to_speech(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()


# **Custom Styling for Streamlit**
def apply_custom_styles():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #ff1e1e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_styles()

# **App Title**
st.title("📰 NewsIntelGraph: AI-Powered Insight Analyzer")

# **Sidebar Theme Toggle**
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

# **AI Chatbot for News**
st.sidebar.subheader("🤖 Ask the News Chatbot")
user_query = st.sidebar.text_input("Enter your news-related question:")


def get_chatbot_response(query):
    """Fetch response from Gemini AI chatbot."""
    if query:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(query)
            return response.text if hasattr(response, 'text') else "Error in fetching response."
        except Exception as e:
            return f"Error: {str(e)}"
    return ""


if user_query:
    chatbot_response = get_chatbot_response(user_query)
    st.sidebar.write("🗣️ Chatbot Response:", chatbot_response)

# **User Input for News Search**
st.sidebar.subheader("🔍 Search News")
topic = st.sidebar.text_input("Enter a news topic:")

# **Entity Frequency Counter**
entity_freq = Counter()

if topic:
    search_url = f'https://www.bing.com/news/search?q={topic.replace(" ", "+")}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raises an error for HTTP failures
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('a', class_='title')[:5]

        if not articles:
            st.error("No news articles found! Try another topic.")
        else:
            st.subheader("📰 Top Articles:")
            full_text = ""
            article_sentiments = []

            for idx, article in enumerate(articles):
                title = article.text
                link = article['href']
                st.markdown(f"[{idx + 1}. {title}]({link})")
                full_text += title + " "

                # **Sentiment Analysis**
                blob = TextBlob(title)
                sentiment_score = blob.sentiment.polarity
                article_sentiments.append(sentiment_score)

                # **Named Entity Recognition**
                doc = nlp(title)
                for ent in doc.ents:
                    entity_freq[ent.text] += 1

            # **Sentiment Analysis Summary**
            sentiment_score = np.mean(article_sentiments)
            st.subheader("📊 Sentiment Analysis:")
            if sentiment_score > 0:
                st.success("Overall Sentiment: Positive 😊")
            elif sentiment_score < 0:
                st.error("Overall Sentiment: Negative 😞")
            else:
                st.warning("Overall Sentiment: Neutral 😐")

            # **Bias Detection**
            subjectivity_score = np.mean([TextBlob(title).sentiment.subjectivity for title in full_text.split('.')])
            st.subheader("🛑 Bias Detection:")
            st.write(f"Estimated Subjectivity: {subjectivity_score:.2f}")

            # **Keyword & Hashtag Generator**
            kw_extractor = yake.KeywordExtractor()
            keywords = kw_extractor.extract_keywords(full_text)
            st.subheader("🔑 Keywords & Hashtags:")
            st.write([kw[0] for kw in keywords[:10]])

            # **Word Cloud**
            if full_text.strip():
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='black' if theme == "Dark" else 'white'
                ).generate(full_text)

                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.subheader("☁️ Word Cloud of News Headlines:")
                st.pyplot(fig)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")

# **Trending Topics Widget**
st.sidebar.subheader("📈 Trending Topics")
trending_url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_KEY}'

try:
    trending_response = requests.get(trending_url).json()
    if 'articles' in trending_response:
        trending_topics = [article['title'] for article in trending_response['articles'][:5]]
        for topic in trending_topics:
            st.sidebar.write(f"- {topic}")
    else:
        st.sidebar.write("No trending topics found.")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"Error fetching trending topics: {e}")
