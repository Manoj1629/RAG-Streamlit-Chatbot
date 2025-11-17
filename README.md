# RAG Streamlit Chatbot (Demo)

## Overview
This project is a demo Streamlit app that demonstrates:
- Retrieval-Augmented Generation (RAG) using sentence-transformers and FAISS.
- Optional web-search integration (SerpAPI or DuckDuckGo fallback).
- Response modes: Concise vs Detailed.
- Modular structure with configs, models, and utils.

## Setup (VSCode)
1. Create a Python virtual environment and activate it.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API keys via environment variables (recommended):
   ```bash
   export OPENAI_API_KEY='sk-...'
   export SERPAPI_API_KEY='...'
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```
