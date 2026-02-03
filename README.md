# Feedback Analysis Tool

A Streamlit-based tool for analyzing customer feedback and product support tickets using LLMs. Supports both local (Ollama) and cloud-based (Groq) language models.

## Features

- **Customer Feedback Analysis**: Extract top feature requests, most loved features, main complaints, and competitor mentions
- **Support Ticket Analysis**: Identify critical issues, looming issues, and potential fixes
- **Multiple LLM Providers**: Choose between local Ollama or cloud-based Groq
- **File Support**: Upload CSV or Excel (.xlsx) files for analysis

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd feedback-analysis
```

### 2. Create environment file

Create a `.env` file in the project root:

```bash
# For Groq (cloud) - get your key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional: For OpenAI (legacy, not currently used)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment (if not exists)
- Install dependencies
- Start the Streamlit app

### Manual Setup (Alternative)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app_llama3v2.py
```

## Usage

1. Select your LLM provider in the sidebar:
   - **Local (Ollama)**: Requires Ollama running locally with a model installed
   - **Groq (Cloud)**: Requires GROQ_API_KEY in your .env file

2. Choose the analysis type:
   - Customer Feedback Data
   - Product Support Tickets

3. Upload your CSV or Excel file

4. Select which columns to include in the analysis

5. Click "Run Analysis"

## LLM Options

### Ollama (Local)
- Free, runs on your machine
- Requires [Ollama](https://ollama.ai) installed
- Default model: `llama3` (configurable in sidebar)
- Data stays on your machine

### Groq (Cloud)
- Fast inference, requires API key
- Available models: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
- Get your API key at [console.groq.com](https://console.groq.com)

## Project Structure

```
├── app_llama3v2.py     # Main app (recommended)
├── app_llama.py        # Alternative Ollama-only app
├── llama3app_cfa.py    # CFA variant app
├── config.py           # Configuration loader
├── requirements.txt    # Python dependencies
├── setup.sh            # Setup and run script
└── .env                # Your API keys (not in repo)
```

## Requirements

- Python 3.9+
- For local LLM: [Ollama](https://ollama.ai) with a model installed
- For cloud LLM: Groq API key

## History

This project was originally created in June 2024 for analyzing customer feedback using local LLMs. It has since been updated to support cloud-based LLM providers (Groq) for faster inference.

## License

MIT
