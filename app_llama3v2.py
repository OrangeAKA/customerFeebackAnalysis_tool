import streamlit as st
import pandas as pd
import ollama
import config

# Try to import groq, but make it optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def llm_chat(messages, provider="ollama", model=None):
    """
    Unified LLM chat function that supports multiple providers.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        provider: 'ollama' or 'groq'
        model: Model name (defaults vary by provider)
    
    Returns:
        Response text from the LLM
    """
    if provider == "ollama":
        model = model or "llama3"
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package is not installed. Run: pip install groq")
        
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")
        
        model = model or "llama-3.1-70b-versatile"
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'groq'.")


def read_file(file):
    """Read CSV or Excel file and handle header detection."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a .csv or .xlsx file.")
        return None
    
    # Check for missing headers or empty rows at the start
    if df.iloc[0].isnull().all():
        df.columns = df.iloc[1]
        df = df[2:]
    elif df.columns.isnull().all():
        df.columns = df.iloc[0]
        df = df[1:]

    df.reset_index(drop=True, inplace=True)
    return df


def analyze_data(data, selected_columns, analysis_type, provider, model):
    """Analyze data using the selected LLM provider."""
    combined_text = " ".join(data[selected_columns].astype(str).apply(lambda x: ' '.join(x), axis=1))
    
    if analysis_type == "Customer Feedback Data":
        prompt = (
            f"Analyze the following customer feedback and extract the following information:\n"
            f"1. Top feature requests\n"
            f"2. Most loved features\n"
            f"3. Main complaints\n"
            f"4. Competitor mentions\n\n"
            f"Customer feedback:\n{combined_text}"
        )
    else:
        prompt = (
            f"Analyze the following product support tickets and extract the following information:\n"
            f"1. Critical issues\n"
            f"2. Looming issues\n"
            f"3. Potential fixes for issues\n\n"
            f"Support tickets:\n{combined_text}"
        )
    
    messages = [
        {"role": "system", "content": "You are an expert data analyst."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = llm_chat(messages, provider=provider, model=model)
        if response:
            return response.strip()
    except Exception as e:
        st.error(f"Error calling LLM: {str(e)}")
    return None


def extract_section(text, section_title):
    """Extract a section from the analysis text based on title."""
    lines = text.split('\n')
    start_index = next((i for i, line in enumerate(lines) if section_title.lower() in line.lower()), None)
    if start_index is None:
        return "No data found"
    
    end_index = next(
        (i for i, line in enumerate(lines[start_index + 1:], start=start_index + 1) if line.strip() == ""),
        len(lines)
    )
    return "\n".join(lines[start_index + 1:end_index]).strip()


def display_analysis(analysis_text, analysis_type):
    """Display the analysis results."""
    st.text_area("Full Analysis", value=analysis_text, height=300)

    if analysis_type == "Customer Feedback Data":
        st.subheader("Top Feature Requests")
        st.text_area("Top Feature Requests", value=extract_section(analysis_text, "Top feature requests"), height=150)

        st.subheader("Most Loved Features")
        st.text_area("Most Loved Features", value=extract_section(analysis_text, "Most loved features"), height=150)

        st.subheader("Main Complaints")
        st.text_area("Main Complaints", value=extract_section(analysis_text, "Main complaints"), height=150)

        st.subheader("Competitor Mentions")
        st.text_area("Competitor Mentions", value=extract_section(analysis_text, "Competitor mentions"), height=150)
    
    elif analysis_type == "Product Support Tickets":
        st.subheader("Critical Issues")
        st.text_area("Critical Issues", value=extract_section(analysis_text, "Critical issues"), height=150)

        st.subheader("Looming Issues")
        st.text_area("Looming Issues", value=extract_section(analysis_text, "Looming issues"), height=150)

        st.subheader("Potential Fixes")
        st.text_area("Potential Fixes", value=extract_section(analysis_text, "Potential fixes"), height=150)


# Streamlit UI
st.title("Feedback and Support Ticket Analysis Tool")

# Sidebar for LLM configuration
st.sidebar.header("LLM Configuration")

# Provider selection
provider_options = ["Local (Ollama)"]
if GROQ_AVAILABLE and config.GROQ_API_KEY:
    provider_options.append("Groq (Cloud)")

provider_display = st.sidebar.selectbox(
    "Select LLM Provider",
    provider_options,
    help="Choose between local Ollama or cloud-based Groq"
)

# Map display name to internal provider name
provider = "ollama" if "Ollama" in provider_display else "groq"

# Model selection based on provider
if provider == "ollama":
    model = st.sidebar.text_input(
        "Ollama Model",
        value="llama3",
        help="Enter the name of your local Ollama model (e.g., llama3, mistral, codellama)"
    )
else:
    groq_models = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]
    model = st.sidebar.selectbox(
        "Groq Model",
        groq_models,
        help="Select a Groq model"
    )

# Show provider status
if provider == "ollama":
    st.sidebar.info("Using local Ollama. Make sure Ollama is running.")
else:
    st.sidebar.success("Using Groq cloud API.")

# Main content
file_type = st.radio("Select file type", ["Customer Feedback Data", "Product Support Tickets"])

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("Dataset Preview:", df.head())

        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to include in analysis",
            all_columns,
            default=all_columns
        )

        if st.button("Run Analysis"):
            with st.spinner(f"Running analysis using {provider_display}..."):
                analysis_text = analyze_data(df, selected_columns, file_type, provider, model)
            
            if analysis_text:
                st.success("Analysis completed!")
                display_analysis(analysis_text, file_type)
            else:
                st.error("Analysis failed. Please check your LLM configuration.")
