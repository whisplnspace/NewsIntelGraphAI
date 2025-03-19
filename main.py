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
import os
import google.generativeai as genai
from collections import Counter
import pandas as pd

# ✅ **Load API Keys from Environment Variables**
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not GEMINI_API_KEY or not NEWSAPI_KEY:
    st.error("⚠️ API Keys not found. Please set environment variables.")

# ✅ **Configure Gemini AI**
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ✅ **Download NLTK Dependencies**
nltk.download('punkt')

# ✅ **Load SpaCy NLP Model (Cached)**
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_lg")

nlp = load_spacy()

# ✅ **Custom Styling for Streamlit**
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

# ✅ **App Title**
st.title("📰 NewsIntelGraph: AI-Powered Insight Analyzer")

# ✅ **Sidebar Theme Toggle**
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

# ✅ **AI Chatbot for News**
st.sidebar.subheader("🤖 Ask the News Chatbot")
user_query = st.sidebar.text_input("Enter your news-related question:")

def get_chatbot_response(query):
    """Fetch response from Gemini AI chatbot."""
    if query and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(query)
            return response.text if hasattr(response, 'text') else "⚠️ Error fetching response."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    return ""

if user_query:
    chatbot_response = get_chatbot_response(user_query)
    st.sidebar.write("🗣️ Chatbot Response:", chatbot_response)

# ✅ **User Input for News Search**
st.sidebar.subheader("🔍 Search News")
topic = st.sidebar.text_input("Enter a news topic:")

# ✅ **Entity Frequency Counter**
entity_freq = Counter()

if topic:
    try:
        # ✅ **Fetch News Using NewsAPI**
        news_url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWSAPI_KEY}"
        response = requests.get(news_url)
        data = response.json()

        if "articles" not in data or not data["articles"]:
            st.error("⚠️ No news articles found! Try another topic.")
        else:
            st.subheader("📰 Top Articles:")
            full_text = ""
            article_sentiments = []

            for idx, article in enumerate(data["articles"][:5]):
                title = article.get("title", "No Title")
                link = article.get("url", "#")
                st.markdown(f"[{idx + 1}. {title}]({link})")
                full_text += title + " "

                # ✅ **Sentiment Analysis**
                blob = TextBlob(title)
                sentiment_score = blob.sentiment.polarity
                article_sentiments.append(sentiment_score)

                # ✅ **Named Entity Recognition**
                doc = nlp(title)
                for ent in doc.ents:
                    entity_freq[ent.text] += 1

            # ✅ **Sentiment Analysis Summary**
            sentiment_score = np.mean(article_sentiments) if article_sentiments else 0
            st.subheader("📊 Sentiment Analysis:")
            if sentiment_score > 0:
                st.success("Overall Sentiment: Positive 😊")
            elif sentiment_score < 0:
                st.error("Overall Sentiment: Negative 😞")
            else:
                st.warning("Overall Sentiment: Neutral 😐")

            # ✅ **Bias Detection**
            subjectivity_scores = [TextBlob(title).sentiment.subjectivity for title in full_text.split('.') if title]
            subjectivity_score = np.mean(subjectivity_scores) if subjectivity_scores else 0
            st.subheader("🛑 Bias Detection:")
            st.write(f"Estimated Subjectivity: {subjectivity_score:.2f}")

            # ✅ **Keyword & Hashtag Generator**
            kw_extractor = yake.KeywordExtractor()
            keywords = kw_extractor.extract_keywords(full_text)
            st.subheader("🔑 Keywords & Hashtags:")
            st.write([kw[0] for kw in keywords[:10]])

            # ✅ **Word Cloud**
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
        st.error(f"⚠️ Error fetching news: {e}")

# ✅ **Trending Topics Widget**
st.sidebar.subheader("📈 Trending Topics")
try:
    trending_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_KEY}"
    trending_response = requests.get(trending_url).json()

    if 'articles' in trending_response and trending_response['articles']:
        trending_topics = [article.get('title', 'No Title') for article in trending_response['articles'][:5]]
        for topic in trending_topics:
            st.sidebar.write(f"- {topic}")
    else:
        st.sidebar.write("No trending topics found.")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"⚠️ Error fetching trending topics: {e}")
