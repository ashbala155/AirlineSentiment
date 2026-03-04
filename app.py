import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Airline Sentiment Dashboard (Simulated)",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------
# SENTIMENT LEXICON
# ---------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "amazing", "happy",
    "awesome", "smooth", "fantastic", "nice", "friendly"
}
NEGATIVE_WORDS = {
    "bad", "terrible", "worst", "hate", "awful", "delay",
    "late", "cancelled", "angry", "poor", "rude"
}

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv("posts.csv", parse_dates=["created_at"])
    df["combined_text"] = df["title"] + " " + df["text"]
    return df

def analyze_sentiment(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    words = text.split()
    pos_count = sum(word in POSITIVE_WORDS for word in words)
    neg_count = sum(word in NEGATIVE_WORDS for word in words)
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

def generate_wordcloud(text_series):
    text = " ".join(text_series)
    wordcloud = WordCloud(
        width=800, height=400, stopwords=STOPWORDS, background_color="white"
    ).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Controls")
sample_size = st.sidebar.slider("Number of posts to simulate live", 5, 50, 20)
refresh = st.sidebar.button("🔄 Refresh (simulate new posts)")

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
st.title("✈️ Airline Sentiment Dashboard (Simulated Live)")

# Load data
data = load_data()

# Simulate "live" posts
simulated_posts = data.sample(sample_size, replace=False, random_state=np.random.randint(0,10000))
simulated_posts["sentiment"] = simulated_posts["combined_text"].apply(analyze_sentiment)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Posts", len(simulated_posts))
col2.metric("Positive", (simulated_posts["sentiment"] == "positive").sum())
col3.metric("Negative", (simulated_posts["sentiment"] == "negative").sum())

# ---------------------------------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------------------------------
st.subheader("Sentiment Distribution")
sentiment_counts = simulated_posts["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]
fig_bar = px.bar(
    sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title="Sentiment Breakdown"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# TIME DISTRIBUTION
# ---------------------------------------------------
st.subheader("Post Activity Over Time")
simulated_posts["hour"] = simulated_posts["created_at"].dt.hour
time_dist = simulated_posts.groupby("hour").size().reset_index(name="count")
fig_time = px.line(
    time_dist, x="hour", y="count", markers=True, title="Posts by Hour"
)
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------
# WORD CLOUD
# ---------------------------------------------------
st.subheader("Word Cloud by Sentiment")
selected_sentiment = st.selectbox("Choose Sentiment", ["positive", "neutral", "negative"])
filtered_text = simulated_posts[simulated_posts["sentiment"] == selected_sentiment]["combined_text"]
if not filtered_text.empty:
    fig_wc = generate_wordcloud(filtered_text)
    st.pyplot(fig_wc)
else:
    st.info("No posts for selected sentiment.")

# ---------------------------------------------------
# RAW DATA
# ---------------------------------------------------
with st.expander("View Raw Data"):
    st.dataframe(simulated_posts)
