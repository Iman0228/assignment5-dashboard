import streamlit as st
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
from python_speech_features import mfcc, delta
# Audio model from Assignment 4
def load_audio_model():
    model = nn.Sequential()
    model.add_module("net", nn.Sequential(
        nn.Linear(26, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    ))
    model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
    model.eval()
    return model
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
