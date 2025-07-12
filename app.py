# app.py
import streamlit as st

# Import necessary libraries for each project
import tensorflow as tf
import numpy as np
import cv2
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
import PyPDF2
import io  # For handling uploaded files
from typing import Optional
import requests
import openai
import os

# Robust OpenAI API key loading
openai_api_key = None
try:
    openai_api_key = st.secrets["openai_api_key"]
except Exception:
    openai_api_key = os.environ.get("openai_api_key")

if not openai_api_key:
    st.sidebar.error("OpenAI API key not found. Please set 'openai_api_key' in Streamlit secrets or your .env file.")

# Use OpenAI v1.x client
client = openai.OpenAI(api_key=openai_api_key)

# Set up the Streamlit app
st.title("KH AI Project Demos")
st.markdown("This dashboard showcases three example AI projects tailored for KH's services. Each tab demonstrates a different AI application using various vendors.")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Traffic Sign Condition Assessment",
    "Traffic Congestion Prediction",
    "EV Charger Site Feasibility Analysis",
])

with tab1:
    st.header("Project 1: Computer Vision for Traffic Sign Condition Assessment")
    st.markdown("""
    **Vendor/Option:** Hugging Face Transformers  
    **Description:** Automatically detect and assess traffic sign conditions from images.  
    **Improves Clients:** Faster asset maintenance for transportation projects.  
    Upload an image to detect signs and assess their condition.
    """)
    
    # Upload image
    uploaded_image = st.file_uploader("Upload a traffic sign image (JPG/PNG)", type=["jpg", "png", "jpeg"], key="tab1_uploader")
    
    if uploaded_image:
        try:
            # Use Hugging Face image classification pipeline
            classifier = pipeline("image-classification", model="microsoft/resnet-50")
            
            # Process the image
            image_bytes = uploaded_image.read()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB for the model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Run classification
            results = classifier(pil_image)
            
            # Display results
            st.image(image_rgb, caption="Uploaded Image", channels="RGB")
            
            st.subheader("Classification Results")
            for i, result in enumerate(results[:5]):  # Show top 5 results
                st.write(f"{i+1}. {result['label']} (confidence: {result['score']:.3f})")
            
            # Simple condition assessment based on confidence
            top_confidence = results[0]['score']
            if top_confidence > 0.8:
                condition = "Sign is in good condition."
            elif top_confidence > 0.6:
                condition = "Sign needs review."
            else:
                condition = "You need to have the sign replaced."
            
            st.subheader("Condition Assessment")
            st.write(f"Overall condition: {condition}")
            st.write(f"Top classification confidence: {top_confidence:.3f}")
            
        except Exception as e:
            st.error(f"Error running classification: {e}")
            st.info("This demo uses a general image classification model. For specific traffic sign detection, you would need a custom-trained model.")

with tab2:
    st.header("Project 2: Machine Learning for Traffic Congestion Prediction")
    st.markdown("""
    **Vendor/Option:** Meta PyTorch  
    **Description:** Predict traffic congestion using time-series data.  
    **Improves Clients:** Better planning for roadways and signals.  
    Upload a CSV with 'congestion_level' column or use sample data.
    """)
    
    # Sample data generator or upload
    sample_data = pd.DataFrame({'congestion_level': np.random.uniform(0, 100, 100)})
    data_option = st.radio("Data Source", ("Use Sample Data", "Upload CSV"), key="tab2_radio")
    
    if data_option == "Upload CSV":
        uploaded_csv = st.file_uploader("Upload traffic data CSV", type="csv", key="tab2_uploader")
        if uploaded_csv:
            data = pd.read_csv(uploaded_csv)
        else:
            data = sample_data
    else:
        data = sample_data
    
    if st.button("Run Prediction", key="tab2_button"):
        # LSTM Model definition
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
                super().__init__()
                self.hidden_layer_size = hidden_layer_size
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.linear = nn.Linear(hidden_layer_size, output_size)
            
            def forward(self, input_seq):
                # input_seq shape: (batch_size, seq_len, input_size)
                lstm_out, _ = self.lstm(input_seq)
                # Take the last output from the sequence
                last_output = lstm_out[:, -1, :]
                predictions = self.linear(last_output)
                return predictions
        
        scaler = MinMaxScaler()
        data['congestion_level'] = scaler.fit_transform(data[['congestion_level']])
        
        # Prepare training data
        seq_length = 10
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data['congestion_level'].values[i:i+seq_length])
            y.append(data['congestion_level'].values[i+seq_length])
        
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add input dimension
        y = torch.tensor(y, dtype=torch.float32)
        
        model = LSTMModel()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Simple training loop
        for epoch in range(10):  # More epochs for better training
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_function(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
        
        # Make prediction
        test_seq = torch.tensor(data['congestion_level'].values[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        predicted_congestion = model(test_seq)
        predicted_value = scaler.inverse_transform([[predicted_congestion.item()]])[0][0]
        st.write(f'Predicted congestion: {predicted_value:.2f}')
        
        # Show some training info
        st.write(f'Training completed with {len(X)} samples')
        st.write(f'Final loss: {loss.item():.4f}')

with tab3:
    st.header("Project 3: Natural Language Processing for EV Charger Site Feasibility Analysis")
    st.markdown("""
    **Vendor/Option:** Hugging Face Transformers  
    **Description:** Analyze documents for EV site feasibility.  
    **Improves Clients:** Quicker permitting and site selection.  
    Upload a PDF report to summarize and classify.
    """)
    
    uploaded_pdf = st.file_uploader("Upload site feasibility PDF", type="pdf", key="tab3_uploader")
    
    if uploaded_pdf:
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        
        # Read PDF
        pdf_bytes = uploaded_pdf.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        
        # Summarize
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)
        
        # Classify
        labels = ['feasible for EV charger', 'zoning restricted', 'environmental impact high', 'permitting required']
        result = classifier(text[:512], candidate_labels=labels)
        st.subheader("Classification")
        st.write(f"Top: {result['labels'][0]} (score: {result['scores'][0]:.2f})")

# --- SIDEBAR CHATBOT ---
st.sidebar.header("AI Chatbot (Multi-Modal Demo)")
st.sidebar.markdown("""
Chat with the AI and upload files. The bot will use OpenAI to decide whether to answer directly or use one of the analysis tools (image, CSV, or PDF).
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

API_URL = "http://localhost:8000"  # Update if deploying elsewhere

# Tool definitions for OpenAI function calling
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            "description": "Classify a traffic sign image and assess its condition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "The name of the image file."}
                },
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_congestion",
            "description": "Predict traffic congestion from a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "The name of the CSV file."}
                },
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pdf",
            "description": "Summarize and classify an EV site feasibility PDF report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "The name of the PDF file."}
                },
                "required": ["file_name"]
            }
        }
    }
]

def call_api_tool(endpoint, files):
    try:
        response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

# Chatbot UI in sidebar
user_input = st.sidebar.chat_input("Ask me to analyze an image, CSV, or PDF!")

# Determine if the assistant is waiting for a file
waiting_for_file = False
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "assistant" and "upload" in last_msg["content"].lower():
        waiting_for_file = True

# Show a highlighted prompt if waiting for a file
if waiting_for_file:
    st.sidebar.info("Please upload your file below to continue.", icon="📎")

# Always show the file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (image, CSV, or PDF)",
    type=["jpg", "jpeg", "png", "csv", "pdf"],
    key="chatbot_uploader"
)
if uploaded_file:
    st.session_state.uploaded_files[uploaded_file.name] = uploaded_file

# 1. Handle user input and file upload, and append to chat history
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
if uploaded_file and not user_input:
    st.session_state.chat_history.append({"role": "user", "content": f"I have uploaded a file named {uploaded_file.name}."})

# 2. If there was new input, call OpenAI/tools and append assistant response
if user_input or uploaded_file:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. If the user asks to analyze an image, CSV, or PDF, use the appropriate tool. Use the file name provided by the user or the most recently uploaded file. If a required file is missing, ask the user to upload it."}
    ]
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    response = None
    tool_result = None
    try:
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
        ai_message = completion.choices[0].message
        # Check if OpenAI wants to call a tool
        if ai_message.tool_calls:
            tool_call = ai_message.tool_calls[0]
            tool_name = tool_call.function.name
            import json
            tool_args = json.loads(tool_call.function.arguments)
            file_name = tool_args.get("file_name")
            file_obj = st.session_state.uploaded_files.get(file_name)
            if not file_obj:
                tool_result = f"Please upload the required file named '{file_name}' to continue."
                # Prompt user to upload file
                show_file_uploader = True
            else:
                if tool_name == "classify_image":
                    files = {"file": (file_obj.name, file_obj, file_obj.type)}
                    api_result = call_api_tool("/classify-image", files)
                    if "error" in api_result:
                        tool_result = f"Error: {api_result['error']}"
                    else:
                        tool_result = f"Image classified as: {api_result['label']} (confidence: {api_result['confidence']:.3f}). {api_result['condition']}"
                elif tool_name == "predict_congestion":
                    files = {"file": (file_obj.name, file_obj, file_obj.type)}
                    api_result = call_api_tool("/predict-congestion", files)
                    if "error" in api_result:
                        tool_result = f"Error: {api_result['error']}"
                    else:
                        tool_result = f"Predicted congestion: {api_result['predicted_congestion']:.2f}"
                elif tool_name == "analyze_pdf":
                    files = {"file": (file_obj.name, file_obj, file_obj.type)}
                    api_result = call_api_tool("/analyze-pdf", files)
                    if "error" in api_result:
                        tool_result = f"Error: {api_result['error']}"
                    else:
                        tool_result = f"Summary: {api_result['summary']}\nTop classification: {api_result['top_classification']} (score: {api_result['score']:.2f})"
                else:
                    tool_result = f"Unknown tool: {tool_name}"
            response = tool_result
        else:
            response = ai_message.content
    except Exception as e:
        response = f"Error communicating with OpenAI: {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# 3. Render chat history (after all appends)
for msg in st.session_state.chat_history:
    with st.sidebar:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])