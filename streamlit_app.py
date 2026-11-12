import streamlit as st
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
from python_speech_features import mfcc, delta
st.title("Customer Insights")

st.write("Use this dashboard to explore customer churn and the sentiment of audio clips.")
churn_tab, audio_tab = st.tabs(["Customer Churn", "Audio Sentiment"])

with churn_tab:
    st.header("Customer Churn")

with audio_tab:
    st.header("Audio Sentiment")
    audio_file = st.file_uploader("Choose an audio clip", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file)
