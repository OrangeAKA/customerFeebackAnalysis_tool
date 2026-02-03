import os
from dotenv import load_dotenv

load_dotenv()

# Try to get from Streamlit secrets first (for Streamlit Cloud deployment)
# Fall back to environment variables (for local development)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
