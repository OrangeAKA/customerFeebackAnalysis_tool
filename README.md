# Feedback Analysis Tool

AI-powered customer feedback analysis with actionable insights for Product Managers and Operations teams.

## Features

### Core Analysis
- **Row-Level Tagging**: Each feedback item is automatically tagged with:
  - Category (feature request, bug report, complaint, praise, etc.)
  - Sentiment (positive, negative, neutral)
  - Priority (high, medium, low)
  - One-sentence summary

### Visual Dashboard
- Category breakdown (pie chart)
- Sentiment distribution (bar chart)
- Priority matrix
- Interactive data table with filtering

### Executive Summary
- Key metrics at a glance
- Top findings with specific numbers
- AI-generated actionable recommendations
- High-priority items highlighted

### Trend Analysis
- Upload multiple datasets to compare over time
- Category trends visualization
- Sentiment trends over periods
- Period-over-period delta indicators

### Root Cause Analysis (CrewAI Agents)
- Multi-agent AI system for deeper insights
- Upload product documentation and release notes
- Automatic correlation of feedback with releases
- Documentation gap identification
- Executive report with prioritized recommendations

### Export Options
- Tagged data as CSV
- Structured JSON export
- Markdown report for sharing

### LLM Providers
- **Groq (Cloud)**: Fast inference with DeepSeek and Llama models
- **Ollama (Local)**: Privacy-focused, runs on your machine

## Quick Start

### Option 1: Streamlit Cloud (Recommended)

The app is deployed at: [Your Streamlit Cloud URL]

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/OrangeAKA/customerFeebackAnalysis_tool.git
cd customerFeebackAnalysis_tool

# Create environment file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 3: Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app_llama3v2.py
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required for Groq (cloud) provider
GROQ_API_KEY=your_groq_api_key_here

# Optional
OPENAI_API_KEY=your_openai_api_key_here
```

Get your Groq API key at [console.groq.com](https://console.groq.com)

### Streamlit Cloud Deployment

1. Fork/clone this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repo
4. Add `GROQ_API_KEY` in Settings > Secrets

## Usage

### Single Analysis Mode

1. Select "Single Analysis" mode
2. Choose your LLM provider and model
3. Upload a CSV or Excel file with feedback data
4. Select the column containing feedback text
5. Click "Run Analysis"
6. Explore results in the tabs:
   - Executive Summary
   - Visual Dashboard
   - Tagged Data (with filters)
   - Export

### Trend Analysis Mode

1. Select "Trend Analysis" mode
2. Upload 2+ datasets representing different time periods
3. Label each time period
4. Click "Run Trend Analysis"
5. View category and sentiment trends over time

## Supported Models

### Groq (Cloud)
| Model | Best For |
|-------|----------|
| llama-3.3-70b-versatile | Best quality, recommended default |
| llama-3.1-8b-instant | Fast, cost-effective |
| openai/gpt-oss-120b | OpenAI's flagship open model |
| openai/gpt-oss-20b | OpenAI's smaller open model |

### Ollama (Local)
Any model installed locally (llama3, mistral, etc.)

## Project Structure

```
├── app_llama3v2.py          # Main application
├── config.py                # Configuration loader
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup and run script
├── agents/                  # CrewAI multi-agent system
│   ├── __init__.py
│   ├── crew_setup.py        # Agent definitions
│   ├── tools.py             # ChromaDB retrieval tools
│   └── document_processor.py # PDF/MD parsing & indexing
├── .streamlit/
│   └── secrets.toml.example # Secrets template for deployment
├── deprecated/              # Legacy app versions
│   ├── app_llama.py
│   ├── llama3app_cfa.py
│   └── README.md
└── README.md
```

## Requirements

- Python 3.9+
- For cloud LLM: Groq API key
- For local LLM: [Ollama](https://ollama.ai) installed

## History

This project was originally created in June 2024 for analyzing customer feedback using local LLMs. It has since been enhanced with:
- Cloud-based LLM support (Groq)
- Row-level tagging and categorization
- Visual dashboards
- Executive summaries with AI recommendations
- Trend analysis across time periods
- Export functionality
- **CrewAI multi-agent orchestration** for root cause analysis with RAG (Feb 2026)

## License

MIT
