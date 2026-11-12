# assignment5 dashboard
# Customer Insights Dashboard

I used my models from Assignments 3 and 4 to build this dashboard. The fraud detection tab checks transaction details for possible fraud, and the audio tab predicts whether a clip sounds positive or negative. These tools could help a business review transactions and customer feedback, but the results still need to be checked to be on the safe side 

# How to run

Keep all files in the same folder. Open a terminal in that folder and run:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

# Fraud Detection tab

The chart shows how the model’s fraud score changes when the transaction amount changes. All the other transaction details stay the same, so we can focus on how the amount affects the result. In the example I tested, the score generally went down as the amount increased. This shows how my model responds to these inputs, but it does not mean larger transactions are always safer. A business could use the chart to better understand the prediction before deciding whether a transaction needs a closer look.

# Audio Sentiment tab

Upload a mono WAV audio clip, which means a clip with one audio channel. You can listen to it and see whether the model predicts Negative or Positive sentiment.

# Included files

- streamlit_app.py: runs the dashboard
- requirements.txt: lists the packages needed to run the app
- audio_model.pth and fraud_model.pth: contain the trained models
- audio_settings.npz and fraud_settings.npz: contain saved values used to prepare the inputs
- README.md: explains how to run and use the app
