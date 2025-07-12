from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Image classification endpoint
@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    try:
        from transformers import pipeline
        import cv2
        import numpy as np
        from PIL import Image
        classifier = pipeline("image-classification", model="microsoft/resnet-50")
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        results = classifier(pil_image)
        top_confidence = results[0]['score']
        if top_confidence > 0.8:
            condition = "Sign is in good condition."
        elif top_confidence > 0.6:
            condition = "Sign needs review."
        else:
            condition = "You need to have the sign replaced."
        return {"label": results[0]['label'], "confidence": top_confidence, "condition": condition}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Congestion prediction endpoint
@app.post("/predict-congestion")
async def predict_congestion(file: UploadFile = File(...)):
    try:
        import pandas as pd
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = pd.read_csv(file.file)
        data['congestion_level'] = scaler.fit_transform(data[['congestion_level']])
        seq_length = 10
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data['congestion_level'].values[i:i+seq_length])
            y.append(data['congestion_level'].values[i+seq_length])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
                super().__init__()
                self.hidden_layer_size = hidden_layer_size
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.linear = nn.Linear(hidden_layer_size, output_size)
            def forward(self, input_seq):
                lstm_out, _ = self.lstm(input_seq)
                last_output = lstm_out[:, -1, :]
                predictions = self.linear(last_output)
                return predictions
        model = LSTMModel()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(10):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
        test_seq = torch.tensor(data['congestion_level'].values[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        predicted_congestion = model(test_seq)
        predicted_value = scaler.inverse_transform([[predicted_congestion.item()]])[0][0]
        return {"predicted_congestion": float(predicted_value)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# PDF analysis endpoint
@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        from transformers import pipeline
        import PyPDF2
        import io
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        pdf_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        labels = ['feasible for EV charger', 'zoning restricted', 'environmental impact high', 'permitting required']
        result = classifier(text[:512], candidate_labels=labels)
        return {"summary": summary, "top_classification": result['labels'][0], "score": result['scores'][0]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 