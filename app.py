import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Airline Sentiment Dashboard (Simulated Live Tweets)",
    page_icon="✈️",
    layout="wide"
)

DATA_URL = "Tweets.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(DATA_URL)
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    df['combined_text'] = df['text']  # keep original text
    return df

data = load_data()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.title("Controls")

sample_size = st.sidebar.slider("Number of tweets to simulate live", 5, 50, 20)
refresh = st.sidebar.button("🔄 Refresh")

# -----------------------------
# SIMULATE LIVE TWEETS
# -----------------------------
simulated_data = data.sample(sample_size, replace=False, random_state=np.random.randint(0,10000))

# -----------------------------
# SHOW RANDOM TWEET
# -----------------------------
st.sidebar.subheader("Show random tweet")
random_sentiment = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
random_tweet_text = simulated_data.query("airline_sentiment == @random_sentiment")[["text"]]
if not random_tweet_text.empty:
    st.sidebar.markdown(random_tweet_text.sample(n=1).iat[0, 0])
else:
    st.sidebar.markdown("No tweets available for this sentiment.")

# -----------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------
st.sidebar.subheader("Number of tweets by sentiment")
select_plot = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='sentiment_viz')
sentiment_count = simulated_data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Hide Sentiment Plot", False):
    st.markdown("### Number of tweets by sentiment")
    if select_plot == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

# -----------------------------
# TWEETS BY HOUR
# -----------------------------
st.sidebar.subheader("Tweets by Hour")
hour = st.sidebar.slider("Select Hour", 0, 23)
hour_data = simulated_data[simulated_data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Hide Tweets by Hour", False, key='hour_hide'):
    st.markdown(f"### Tweets between {hour}:00 and {hour+1}:00")
    st.markdown(f"{len(hour_data)} tweets")
    if not hour_data.empty:
        st.map(hour_data)
        if st.sidebar.checkbox("Show Raw Data", False, key='hour_raw'):
            st.write(hour_data)

# -----------------------------
# TOTAL TWEETS PER AIRLINE
# -----------------------------
st.sidebar.subheader("Total tweets per airline")
airline_viz = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='airline_viz')
airline_count = simulated_data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_count = pd.DataFrame({'Airline': airline_count.index, 'Tweets': airline_count.values.flatten()})

if not st.sidebar.checkbox("Hide Airline Plot", False, key='airline_hide'):
    st.subheader("Total tweets per airline")
    if airline_viz == 'Bar plot':
        fig_air = px.bar(airline_count, x='Airline', y='Tweets', color='Tweets', height=500)
    else:
        fig_air = px.pie(airline_count, values='Tweets', names='Airline')
    st.plotly_chart(fig_air)

# -----------------------------
# BREAKDOWN BY SENTIMENT PER AIRLINE (ROBUST)
# -----------------------------
st.sidebar.subheader("Breakdown airline by sentiment")
selected_airlines = st.sidebar.multiselect(
    "Pick airlines", simulated_data['airline'].unique()
)

def get_sentiment_counts(df, airline):
    df_air = df[df['airline'] == airline]
    if df_air.empty:
        return pd.DataFrame({'Sentiment': ['none'], 'Tweets': [0], 'Airline': [airline]})
    counts = df_air['airline_sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Tweets']
    counts['Airline'] = airline
    return counts

if selected_airlines:
    st.subheader("Breakdown by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='breakdown_viz')

    # Combine data for selected airlines
    plot_df = pd.concat([get_sentiment_counts(simulated_data, airline) for airline in selected_airlines], ignore_index=True)

    if breakdown_type == 'Bar plot':
        fig = px.bar(plot_df, x='Sentiment', y='Tweets', color='Airline', barmode='group', height=500)
        st.plotly_chart(fig)
    else:
        # Pie charts per airline
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        specs = [[{'type':'domain'}]*len(selected_airlines)]
        fig = make_subplots(rows=1, cols=len(selected_airlines), specs=specs, subplot_titles=selected_airlines)

        for i, airline in enumerate(selected_airlines):
            df_air = plot_df[plot_df['Airline'] == airline]
            fig.add_trace(
                go.Pie(labels=df_air['Sentiment'], values=df_air['Tweets'], showlegend=(i==0)),
                row=1, col=i+1
            )

        fig.update_layout(height=500, width=300*len(selected_airlines))
        st.plotly_chart(fig)

# -----------------------------
# WORD CLOUD
# -----------------------------
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Select sentiment for word cloud', ('positive', 'neutral', 'negative'))

filtered_wc = simulated_data[simulated_data['airline_sentiment'] == word_sentiment]['text']
if not filtered_wc.empty:
    st.subheader(f"Word Cloud for {word_sentiment} tweets")
    processed_words = ' '.join([word for word in ' '.join(filtered_wc).split()
                                if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()
else:
    st.info(f"No tweets available for {word_sentiment} sentiment.")
