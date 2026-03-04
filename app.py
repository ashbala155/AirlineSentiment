import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import tweepy


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Real-Time Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide"
)

BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]

if not BEARER_TOKEN:
    st.error("Twitter Bearer Token not found. Add it to your .env file.")
    st.stop()

client = tweepy.Client(bearer_token=BEARER_TOKEN)

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
def fetch_tweets(query: str, max_results: int = 100):
    """Fetch recent tweets from Twitter API."""
    try:
        response = client.search_recent_tweets(
            query=f"{query} -is:retweet lang:en",
            max_results=max_results,
            tweet_fields=["created_at"]
        )

        if response.data is None:
            return pd.DataFrame(columns=["text", "created_at"])

        tweets = [
            {
                "text": tweet.text,
                "created_at": tweet.created_at
            }
            for tweet in response.data
        ]

        df = pd.DataFrame(tweets)
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return pd.DataFrame(columns=["text", "created_at"])


def clean_text(text: str) -> str:
    """Basic tweet cleaning."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower()


def analyze_sentiment(text: str) -> str:
    """Simple rule-based sentiment scoring."""
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

search_query = st.sidebar.text_input(
    "Search Keyword",
    value="US Airlines"
)

max_tweets = st.sidebar.slider(
    "Number of Tweets",
    min_value=10,
    max_value=100,
    value=50
)

refresh_button = st.sidebar.button("🔄 Refresh Data")

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

st.title("✈️ Real-Time Airline Sentiment Analysis")
st.markdown("Live Twitter dashboard with custom sentiment analysis.")

if refresh_button:
    st.cache_data.clear()

with st.spinner("Fetching live tweets..."):
    data = fetch_tweets(search_query, max_tweets)

if data.empty:
    st.warning("No tweets found.")
    st.stop()

data["sentiment"] = data["text"].apply(analyze_sentiment)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Tweets", len(data))
col2.metric("Positive", (data["sentiment"] == "positive").sum())
col3.metric("Negative", (data["sentiment"] == "negative").sum())

# ---------------------------------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------------------------------

st.subheader("Sentiment Distribution")

sentiment_counts = data["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig_bar = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    title="Sentiment Breakdown"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# TIME DISTRIBUTION
# ---------------------------------------------------

st.subheader("Tweet Activity Over Time")

data["hour"] = data["created_at"].dt.hour
time_dist = data.groupby("hour").size().reset_index(name="count")

fig_time = px.line(
    time_dist,
    x="hour",
    y="count",
    markers=True,
    title="Tweets by Hour"
)

st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------
# WORD CLOUD
# ---------------------------------------------------

st.subheader("Word Cloud by Sentiment")

selected_sentiment = st.selectbox(
    "Choose Sentiment",
    ["positive", "neutral", "negative"]
)

filtered_text = data[data["sentiment"] == selected_sentiment]["text"]

if not filtered_text.empty:
    fig_wc = generate_wordcloud(filtered_text)
    st.pyplot(fig_wc)
else:
    st.info("No tweets for selected sentiment.")

# ---------------------------------------------------
# RAW DATA
# ---------------------------------------------------

with st.expander("View Raw Data"):
    st.dataframe(data)
