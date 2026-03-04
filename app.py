import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Airline Sentiment Dashboard (Pushshift)",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------
# SIMPLE SENTIMENT LEXICON
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
def fetch_reddit_pushshift(subreddit: str, query: str, size: int = 50):
    """Fetch recent Reddit posts using Pushshift API."""
    url = f"https://api.pushshift.io/reddit/search/submission/"
    params = {
        "subreddit": subreddit,
        "q": query,
        "size": size,
        "sort": "desc",
        "sort_type": "created_utc"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()["data"]
        posts = []
        for item in data:
            posts.append({
                "title": item.get("title", ""),
                "text": item.get("selftext", ""),
                "created_at": datetime.fromtimestamp(item.get("created_utc", 0), tz=timezone.utc)
            })
        df = pd.DataFrame(posts)
        return df
    except Exception as e:
        st.error(f"Error fetching posts: {e}")
        return pd.DataFrame(columns=["title", "text", "created_at"])

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower()

def analyze_sentiment(text: str) -> str:
    text = clean_text(text)
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
        width=800,
        height=400,
        stopwords=STOPWORDS,
        background_color="white"
    ).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Controls")
subreddit = st.sidebar.text_input("Subreddit", value="travel")
search_query = st.sidebar.text_input("Search Keyword", value="airline")
limit = st.sidebar.slider("Number of Posts", 10, 100, 50)
refresh = st.sidebar.button("🔄 Refresh")
if refresh:
    st.cache_data.clear()

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
st.title("✈️ Airline Sentiment Dashboard (Pushshift)")
st.markdown("Real-time-ish Reddit sentiment dashboard using Pushshift API.")

with st.spinner("Fetching Reddit posts..."):
    data = fetch_reddit_pushshift(subreddit, search_query, limit)

if data.empty:
    st.warning("No posts found.")
    st.stop()

data["combined_text"] = data["title"] + " " + data["text"]
data["sentiment"] = data["combined_text"].apply(analyze_sentiment)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Posts", len(data))
col2.metric("Positive", (data["sentiment"] == "positive").sum())
col3.metric("Negative", (data["sentiment"] == "negative").sum())

# ---------------------------------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------------------------------
st.subheader("Sentiment Distribution")
sentiment_counts = data["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]
fig_bar = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title="Sentiment Breakdown")
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# TIME DISTRIBUTION
# ---------------------------------------------------
st.subheader("Post Activity Over Time")
data["hour"] = data["created_at"].dt.hour
time_dist = data.groupby("hour").size().reset_index(name="count")
fig_time = px.line(time_dist, x="hour", y="count", markers=True, title="Posts by Hour")
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------
# WORD CLOUD
# ---------------------------------------------------
st.subheader("Word Cloud by Sentiment")
selected_sentiment = st.selectbox("Choose Sentiment", ["positive", "neutral", "negative"])
filtered_text = data[data["sentiment"] == selected_sentiment]["combined_text"]
if not filtered_text.empty:
    fig_wc = generate_wordcloud(filtered_text)
    st.pyplot(fig_wc)
else:
    st.info("No posts for selected sentiment.")

# ---------------------------------------------------
# RAW DATA
# ---------------------------------------------------
with st.expander("View Raw Data"):
    st.dataframe(data)
