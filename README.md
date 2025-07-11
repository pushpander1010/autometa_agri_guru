# 🌾 AgriGuru - AI Agent for Smart Farming

AgriGuru is an AI-powered LangGraph agent built with [LangChain](https://www.langchain.com/), [Groq LLM](https://console.groq.com/), and [Hugging Face Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), designed to help farmers plan crop cycles, understand soil and weather, and get intelligent, localized answers to their farming questions.

## 🚀 Features

- 📍 **Auto-Detects Location** (via IP) to personalize weather and soil info
- 🌦️ **Seasonal Weather Summary** using historical trends
- 🧪 **Soil Properties** fetched from Indian government APIs
- 🌱 **AI-based Crop Planning** with revised recommendations based on user intent (e.g. "I want to grow wheat")
- 📈 **Mandi Price Lookup** for the top suggested crop
- 📚 **RAG-Powered Question Answering** with fallback to Groq LLM if context is missing
- 🔁 **Feedback Loop** to refine plans interactively
- 🖥️ **Streamlit UI** for farmer-friendly interaction

## 🧠 Powered By

- **LangGraph / LangChain** for agent flow
- **Groq LLM** (LLaMA 4-based) for reasoning & answers
- **Hugging Face Embeddings** for vector search (FAISS)
- **Streamlit** for web-based chat interface

## 🛠️ Folder Structure

.
├── backend.py # LangGraph logic, tools, embeddings, agent logic
├── ui.py # Streamlit UI for farmer interaction
├── tools.py # Weather, soil, mandi price APIs
├── requirements.txt # All dependencies

bash
Copy
Edit

## 🏁 Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/agriguru.git
   cd agriguru
Create virtualenv and install:

bash
Copy
Edit
pip install -r requirements.txt
Set your environment variables (e.g., GROQ_API_KEY, RAPIDAPI_KEY)

Run the app:

bash
Copy
Edit
streamlit run ui.py
✅ Example Inputs
"Which crops should I grow?"

"I want to grow wheat instead"

"What's the mandi price?"

📦 Dependencies
langchain

langgraph

langchain-groq

langchain-huggingface

sentence-transformers

streamlit

requests

faiss-cpu