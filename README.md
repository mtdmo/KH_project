 # KH AI Project Demos

This project showcases three AI-powered demos for KH:
- **Traffic Sign Condition Assessment** (Image Classification)
- **Traffic Congestion Prediction** (Time Series Forecasting)
- **EV Charger Site Feasibility Analysis** (NLP on PDFs)

It features a multi-modal AI chatbot (sidebar) that uses OpenAI to route user queries to the appropriate tool via REST APIs.

---

## Features
- **Streamlit App**: Interactive dashboard with tabs for each demo and a sidebar AI chatbot.
- **FastAPI Server**: REST endpoints for each AI tool (image, CSV, PDF analysis).
- **OpenAI Integration**: Chatbot uses OpenAI (GPT-4) for conversation and tool selection.
- **Secure API Key Handling**: Supports both local `.env` and Streamlit Cloud secrets.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-dir>
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key

#### **Local Development**
- Create a `.env` file in the project root:
  ```
  openai_api_key=sk-...
  ```
- (Optional) Install `python-dotenv` to auto-load `.env`.

#### **Streamlit Community Cloud**
- Go to your app's settings → **Secrets**
- Add:
  ```
  openai_api_key = sk-...
  ```

### 4. Run the FastAPI Server
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## Usage
- Use the main tabs for direct demo interaction.
- Use the **AI Chatbot** in the sidebar for a conversational, multi-modal experience:
  - Ask questions or upload files (image, CSV, PDF)
  - The bot will use OpenAI to decide which tool to use and return results

---

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- streamlit, fastapi, uvicorn, openai, transformers, torch, tensorflow, Pillow, requests, python-dotenv, etc.

---

## Notes
- The image and PDF demos use Hugging Face models (ResNet-50, BART, etc.)
- The chatbot uses OpenAI's function calling to select tools
- For best results, ensure both the FastAPI server and Streamlit app are running

---

## License
MIT (or your preferred license)
