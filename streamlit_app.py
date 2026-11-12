import streamlit as st

st.title("Customer Insights")

st.write("Use this dashboard to explore customer churn and the sentiment of audio clips.")
churn_tab, audio_tab = st.tabs(["Customer Churn", "Audio Sentiment"])

with churn_tab:
    st.header("Customer Churn")

with audio_tab:
    st.header("Audio Sentiment")
    audio_file = st.file_uploader("Choose an audio clip", type=["wav"])
