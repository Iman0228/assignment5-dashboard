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
def prepare_audio(audio_file):
    sample_rate, sound = wavfile.read(audio_file)
    sound = sound.astype(np.float32)

    features = mfcc(sound, samplerate=sample_rate, numcep=13)
    changes = delta(features, 2)
    features = np.concatenate((features, changes), axis=1)
    return features.mean(axis=0)
def predict_audio(audio_file):
    features = prepare_audio(audio_file)
    settings = np.load("audio_settings.npz")
    features = (features - settings["arr_0"]) / settings["arr_1"]

    model = load_audio_model()
    numbers = torch.tensor(features, dtype=torch.float32)
    result = model(numbers).argmax().item()

    return ["Negative", "Positive"][result]
def load_fraud_model():
    model = nn.Sequential()
    model.add_module("net", nn.Sequential(
        nn.Linear(7, 8),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(8, 2)
    ))
    model.load_state_dict(torch.load("fraud_model.pth", map_location="cpu"))
    model.eval()
    return model
def predict_fraud(values):
    settings = np.load("fraud_settings.npz")
    values = np.array(values)
    values = (values - settings["arr_0"]) / settings["arr_1"]

    model = load_fraud_model()
    numbers = torch.tensor(values, dtype=torch.float32)
    result = model(numbers).argmax().item()

    return ["Not flagged", "Possible fraud"][result]
st.title("Customer Insights")


st.write("Review transactions for possible fraud and explore audio sentiment.")
churn_tab, audio_tab = st.tabs(["Fraud Detection", "Audio Sentiment"])


with churn_tab:
    st.header("Fraud Detection")
    reference = st.text_input("Transaction reference")
    amount = st.number_input("Transaction amount", min_value=0.0)
    merchant = st.number_input("Merchant category code", min_value=0, step=1)
    device = st.number_input("Device type code", min_value=0, step=1)
    time_gap = st.number_input("Time since previous transaction", min_value=0.0)
    mean_amount = st.number_input("Average amount of previous 5 transactions", min_value=0.0)
    std_amount = st.number_input("Standard deviation of previous 5 amounts", min_value=0.0)
    count = st.number_input("Previous transaction count (up to 5)", min_value=0, max_value=5, step=1)

with audio_tab:
    st.header("Audio Sentiment")
    audio_file = st.file_uploader("Choose an audio clip", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file)
        audio_file.seek(0)
        prediction = predict_audio(audio_file)
        st.write("Predicted sentiment:", prediction)
