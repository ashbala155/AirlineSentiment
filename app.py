import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ------------------- Custom CSS for fonts & background -------------------
st.markdown("""
<style>
/* Page background */
body {
    background-color: #f5f5f5;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif;
    color: #1f77b4;
}

/* Sidebar background & font */
[data-testid="stSidebar"] {
    background-color: #e8f0fe;
    font-family: 'Verdana', sans-serif;
}

/* Normal text */
body, p, span {
    font-family: 'Verdana', sans-serif;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Load data -------------------
DATA_URL = "Tweets.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

# ------------------- Page title -------------------
st.title("✈️ Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Dashboard")
st.markdown("Analyze sentiments of tweets with interactive charts and word clouds 🐦")
st.sidebar.markdown("Interactive dashboard to explore airline tweet sentiments.")

# ------------------- Sentiment colors -------------------
color_map = {'positive':'#2ca02c', 'neutral':'#7f7f7f', 'negative':'#d62728'}

# ------------------- Sidebar: Random tweet -------------------
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0,0])

# ------------------- Number of tweets by sentiment -------------------
st.sidebar.markdown("### Number of tweets by sentiment")
select_viz = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

if not st.sidebar.checkbox("Hide sentiment chart", False):
    st.markdown("### Number of tweets by sentiment")
    if select_viz == 'Bar plot':
        fig = px.bar(
            sentiment_count, x='Sentiment', y='Tweets', 
            color='Sentiment', color_discrete_map=color_map,
            height=500
        )
    else:
        fig = px.pie(
            sentiment_count, values='Tweets', names='Sentiment',
            color='Sentiment', color_discrete_map=color_map
        )
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Verdana", size=14, color="#333333"),
        title_font=dict(family="Montserrat", size=20, color="#1f77b4")
    )
    st.plotly_chart(fig)

# ------------------- Tweet locations -------------------
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
hour_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Close map", False, key='map_checkbox'):
    st.markdown(f"### Tweet locations between {hour}:00 and {(hour + 1) % 24}:00")
    st.map(hour_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(hour_data)

# ------------------- Tweets per airline -------------------
st.sidebar.subheader("Total number of tweets for each airline")
airline_viz = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='airline_viz')
airline_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_count = pd.DataFrame({'Airline':airline_count.index, 'Tweets':airline_count.values})

if not st.sidebar.checkbox("Close airline chart", False, key='airline_checkbox'):
    if airline_viz == 'Bar plot':
        st.subheader("Total number of tweets per airline")
        fig_air = px.bar(
            airline_count, x='Airline', y='Tweets', color='Tweets', height=500
        )
    else:
        st.subheader("Total number of tweets per airline")
        fig_air = px.pie(
            airline_count, values='Tweets', names='Airline'
        )
    fig_air.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Verdana", size=14, color="#333333"),
        title_font=dict(family="Montserrat", size=20, color="#1f77b4")
    )
    st.plotly_chart(fig_air)

# ------------------- Breakdown by sentiment per airline -------------------
@st.cache_data
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    return pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values})

st.sidebar.subheader("Breakdown of airline by sentiment")
selected_airlines = st.sidebar.multiselect(
    'Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America')
)
if selected_airlines:
    st.subheader("Breakdown of airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot'], key='breakdown_type')
    fig_break = make_subplots(
        rows=1, cols=len(selected_airlines),
        specs=[[{'type':'domain'}]*len(selected_airlines)] if breakdown_type=='Pie chart' else None,
        subplot_titles=selected_airlines
    )
    for i, airline in enumerate(selected_airlines):
        df_sent = plot_sentiment(airline)
        if breakdown_type == 'Bar plot':
            fig_break.add_trace(
                go.Bar(
                    x=df_sent.Sentiment, y=df_sent.Tweets,
                    marker_color=[color_map[s] for s in df_sent.Sentiment],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        else:
            fig_break.add_trace(
                go.Pie(
                    labels=df_sent.Sentiment, values=df_sent.Tweets,
                    marker_colors=[color_map[s] for s in df_sent.Sentiment],
                    showlegend=True
                ),
                row=1, col=i+1
            )
    fig_break.update_layout(height=600, width=300*len(selected_airlines))
    st.plotly_chart(fig_break)

# ------------------- Histogram per airline sentiment -------------------
if selected_airlines:
    st.subheader("Comparison of airlines by sentiment")
    choice_data = data[data.airline.isin(selected_airlines)]
    fig_hist = px.histogram(
        choice_data, x='airline', y='airline_sentiment',
        histfunc='count', color='airline_sentiment',
        facet_col='airline_sentiment', labels={'airline_sentiment':'Tweets'},
        height=600, width=800,
        color_discrete_map=color_map,
    )
    fig_hist.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Verdana", size=14, color="#333333"),
        )
    )
    st.plotly_chart(fig_hist)

# ------------------- Word Cloud -------------------
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close word cloud", False, key='wordcloud_checkbox'):
    st.subheader(f'Word cloud for {word_sentiment} sentiment')
    df_wc = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df_wc['text'])
    processed_words = ' '.join([w for w in words.split() if 'http' not in w and not w.startswith('@') and w != 'RT'])
    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        background_color='white',
        colormap='viridis',
        width=800,
        height=640
    ).generate(processed_words)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
