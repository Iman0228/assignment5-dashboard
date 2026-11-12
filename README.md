# assignment5-dashboard
# Customer Insights Dashboard

This dashboard brings together my models from Assignments 3 and 4. It helps a business check transactions for possible fraud and explore sentiment in customer audio. The predictions can be wrong, so someone should review the results.

## How to run

Keep all files in the same folder. Open a terminal in that folder and run:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Fraud Detection tab

Enter a transaction reference and the seven transaction details, then click Check transaction. Use the same category codes and time units as the training data. The reference is just a label and does not affect the prediction.

The result will say Not flagged or Possible fraud. The chart shows how changing the transaction amount affects the model’s fraud score while the other details stay the same. This helps explain the model, but it does not prove fraud.

## Audio Sentiment tab

Upload a mono WAV audio clip, which means a clip with one audio channel. You can listen to it and see whether the model predicts Negative or Positive sentiment.

## Included files

- streamlit_app.py: runs the dashboard
- requirements.txt: lists the packages needed to run the app
- audio_model.pth and fraud_model.pth: contain the trained models
- audio_settings.npz and fraud_settings.npz: contain saved values used to prepare the inputs
- README.md: explains how to run and use the app
