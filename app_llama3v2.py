import streamlit as st
import pandas as pd
import json
import ollama
import config
from datetime import datetime

# Try to import groq and plotly
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import CrewAI agents
try:
    from agents import DocumentProcessor, FeedbackAnalysisCrew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


# ============================================================================
# LLM Functions
# ============================================================================

def llm_chat(messages, provider="ollama", model=None, json_mode=False):
    """
    Unified LLM chat function that supports multiple providers.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        provider: 'ollama' or 'groq'
        model: Model name (defaults vary by provider)
        json_mode: If True, request JSON response format
    
    Returns:
        Response text from the LLM
    """
    if provider == "ollama":
        model = model or "llama3"
        kwargs = {"model": model, "messages": messages}
        if json_mode:
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        return response['message']['content']
    
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package is not installed. Run: pip install groq")
        
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")
        
        model = model or "llama-3.3-70b-versatile"
        client = Groq(api_key=api_key)
        
        kwargs = {"model": model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'groq'.")


# ============================================================================
# Data Processing Functions
# ============================================================================

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
    if len(df) > 0 and df.iloc[0].isnull().all():
        df.columns = df.iloc[1]
        df = df[2:]
    elif df.columns.isnull().all():
        df.columns = df.iloc[0]
        df = df[1:]

    df.reset_index(drop=True, inplace=True)
    return df


def tag_single_row(text, analysis_type, provider, model):
    """Tag a single feedback row with category, sentiment, and priority."""
    
    if analysis_type == "Customer Feedback Data":
        categories = "feature_request, bug_report, complaint, praise, question, suggestion, other"
    else:
        categories = "critical_bug, performance_issue, usability_issue, feature_request, documentation, other"
    
    prompt = f"""Analyze this feedback and return a JSON object with these exact fields:
{{
    "category": "one of: {categories}",
    "sentiment": "positive, negative, or neutral",
    "priority": "high, medium, or low",
    "summary": "one sentence summary of the key point"
}}

Feedback to analyze:
"{text}"

Return ONLY the JSON object, no other text."""

    messages = [
        {"role": "system", "content": "You are a feedback analyst. Always respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat(messages, provider=provider, model=model, json_mode=True)
        # Try to parse JSON
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                return result
        except:
            pass
        # Return default values if parsing fails
        return {
            "category": "other",
            "sentiment": "neutral", 
            "priority": "medium",
            "summary": text[:100] + "..." if len(text) > 100 else text
        }
    except Exception as e:
        return {
            "category": "error",
            "sentiment": "neutral",
            "priority": "medium", 
            "summary": f"Error: {str(e)[:50]}"
        }


def tag_batch_rows(texts, indices, analysis_type, provider, model):
    """
    Tag multiple feedback rows in a single API call (batch processing).
    
    Args:
        texts: List of feedback texts to analyze
        indices: List of original indices for these texts
        analysis_type: Type of analysis (Customer Feedback or Support Tickets)
        provider: LLM provider
        model: Model name
        
    Returns:
        List of tag dictionaries
    """
    if analysis_type == "Customer Feedback Data":
        categories = "feature_request, bug_report, complaint, praise, question, suggestion, other"
    else:
        categories = "critical_bug, performance_issue, usability_issue, feature_request, documentation, other"
    
    # Build batch prompt
    feedback_list = "\n".join([
        f"[{i}] \"{text[:500]}\"" for i, text in enumerate(texts)
    ])
    
    prompt = f"""Analyze each feedback item below and return a JSON array with one object per item.
Each object must have these exact fields:
- "id": the number in brackets [X]
- "category": one of: {categories}
- "sentiment": positive, negative, or neutral
- "priority": high, medium, or low
- "summary": one sentence summary (max 100 chars)

Feedback items to analyze:
{feedback_list}

Return ONLY a JSON array like: [{{"id": 0, "category": "...", "sentiment": "...", "priority": "...", "summary": "..."}}, ...]"""

    messages = [
        {"role": "system", "content": "You are a feedback analyst. Always respond with valid JSON array only. No explanations."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat(messages, provider=provider, model=model, json_mode=True)
        
        # Parse JSON array
        results = json.loads(response)
        
        # Handle if response is wrapped in an object
        if isinstance(results, dict):
            # Try common wrapper keys
            for key in ['results', 'items', 'feedback', 'data', 'analysis']:
                if key in results and isinstance(results[key], list):
                    results = results[key]
                    break
        
        if not isinstance(results, list):
            raise ValueError("Response is not a list")
        
        # Create lookup by id
        results_by_id = {}
        for item in results:
            if isinstance(item, dict) and 'id' in item:
                results_by_id[item['id']] = item
        
        # Return results in order, with defaults for missing items
        output = []
        for i, text in enumerate(texts):
            if i in results_by_id:
                item = results_by_id[i]
                output.append({
                    "category": item.get("category", "other"),
                    "sentiment": item.get("sentiment", "neutral"),
                    "priority": item.get("priority", "medium"),
                    "summary": item.get("summary", text[:100])
                })
            else:
                # Fallback for missing items
                output.append({
                    "category": "other",
                    "sentiment": "neutral",
                    "priority": "medium",
                    "summary": text[:100] + "..." if len(text) > 100 else text
                })
        
        return output
        
    except Exception as e:
        # On failure, return defaults for all items
        return [
            {
                "category": "other",
                "sentiment": "neutral",
                "priority": "medium",
                "summary": text[:100] + "..." if len(text) > 100 else text
            }
            for text in texts
        ]


def tag_feedback_rows(df, text_column, analysis_type, provider, model, progress_callback=None, 
                      batch_size=15, sample_size=None):
    """
    Tag each row in the dataframe with category, sentiment, and priority.
    
    Args:
        df: DataFrame with feedback data
        text_column: Column name containing feedback text
        analysis_type: Type of analysis
        provider: LLM provider
        model: Model name
        progress_callback: Optional callback for progress updates
        batch_size: Number of rows to process per API call (default: 15)
        sample_size: If set, randomly sample this many rows (default: None = all rows)
    
    Returns:
        DataFrame with added tag columns
    """
    # Apply sampling if requested
    if sample_size and sample_size < len(df):
        # Stratified-ish sampling: ensure we get variety
        sampled_df = df.sample(n=sample_size, random_state=42)
        sampled_indices = sampled_df.index.tolist()
    else:
        sampled_df = df
        sampled_indices = df.index.tolist()
    
    tagged_data = []
    total_rows = len(sampled_df)
    
    # Prepare all texts and handle empty ones
    all_texts = []
    all_indices = []
    empty_results = {}
    
    for idx in sampled_indices:
        row = df.loc[idx]
        text = str(row[text_column])
        
        if not text.strip() or text.lower() == 'nan':
            empty_results[idx] = {
                "category": "empty",
                "sentiment": "neutral",
                "priority": "low",
                "summary": "Empty feedback"
            }
        else:
            all_texts.append(text)
            all_indices.append(idx)
    
    # Process in batches
    batch_results = {}
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_texts))
        
        batch_texts = all_texts[start_idx:end_idx]
        batch_indices = all_indices[start_idx:end_idx]
        
        # Process batch
        try:
            results = tag_batch_rows(batch_texts, batch_indices, analysis_type, provider, model)
            
            for i, idx in enumerate(batch_indices):
                batch_results[idx] = results[i]
                
        except Exception as e:
            # Fallback to individual processing for this batch
            for i, (text, idx) in enumerate(zip(batch_texts, batch_indices)):
                batch_results[idx] = tag_single_row(text, analysis_type, provider, model)
        
        # Update progress
        if progress_callback:
            progress = (batch_num + 1) / total_batches
            progress_callback(progress)
    
    # Combine results
    all_results = {**empty_results, **batch_results}
    
    # Build output dataframe
    for idx in sampled_indices:
        row = df.loc[idx]
        tags = all_results.get(idx, {
            "category": "other",
            "sentiment": "neutral",
            "priority": "medium",
            "summary": ""
        })
        
        tagged_row = row.to_dict()
        tagged_row.update({
            "_category": tags.get("category", "other"),
            "_sentiment": tags.get("sentiment", "neutral"),
            "_priority": tags.get("priority", "medium"),
            "_summary": tags.get("summary", "")
        })
        tagged_data.append(tagged_row)
    
    return pd.DataFrame(tagged_data)


# ============================================================================
# Visualization Functions
# ============================================================================

def display_visualizations(tagged_df):
    """Display visual charts for the tagged data."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Install with: pip install plotly")
        return
    
    st.subheader("Visual Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category breakdown
        category_counts = tagged_df['_category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Feedback by Category",
            hole=0.4
        )
        fig_cat.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiment_counts = tagged_df['_sentiment'].value_counts()
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        fig_sent = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        fig_sent.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Count")
        st.plotly_chart(fig_sent, use_container_width=True)
    
    # Priority breakdown
    priority_counts = tagged_df['_priority'].value_counts()
    priority_order = ['high', 'medium', 'low']
    priority_counts = priority_counts.reindex([p for p in priority_order if p in priority_counts.index])
    
    colors_priority = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#3498db'}
    fig_priority = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        title="Priority Distribution",
        color=priority_counts.index,
        color_discrete_map=colors_priority
    )
    fig_priority.update_layout(showlegend=False, xaxis_title="Priority", yaxis_title="Count")
    st.plotly_chart(fig_priority, use_container_width=True)


# ============================================================================
# Executive Summary Functions
# ============================================================================

def generate_executive_summary(tagged_df):
    """Generate statistics for executive summary."""
    total = len(tagged_df)
    
    # Filter out empty/error categories
    valid_df = tagged_df[~tagged_df['_category'].isin(['empty', 'error'])]
    
    # Category breakdown
    categories = valid_df['_category'].value_counts()
    top_category = categories.index[0] if len(categories) > 0 else "N/A"
    top_category_count = categories.iloc[0] if len(categories) > 0 else 0
    
    # Sentiment breakdown
    sentiment = valid_df['_sentiment'].value_counts()
    negative_count = sentiment.get('negative', 0)
    positive_count = sentiment.get('positive', 0)
    negative_pct = (negative_count / len(valid_df) * 100) if len(valid_df) > 0 else 0
    positive_pct = (positive_count / len(valid_df) * 100) if len(valid_df) > 0 else 0
    
    # Priority items
    high_priority = valid_df[valid_df['_priority'] == 'high']
    
    return {
        'total_items': total,
        'valid_items': len(valid_df),
        'top_category': top_category,
        'top_category_count': top_category_count,
        'negative_pct': negative_pct,
        'positive_pct': positive_pct,
        'negative_count': negative_count,
        'positive_count': positive_count,
        'high_priority_count': len(high_priority),
        'high_priority_items': high_priority['_summary'].tolist()[:5],
        'categories': categories.to_dict(),
        'sentiments': sentiment.to_dict()
    }


def display_executive_summary(summary, tagged_df, provider, model):
    """Display the executive summary."""
    st.subheader("Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Feedback", summary['total_items'])
    col2.metric("Negative Sentiment", f"{summary['negative_pct']:.1f}%", 
                delta=None if summary['negative_pct'] < 30 else "High", 
                delta_color="inverse")
    col3.metric("Positive Sentiment", f"{summary['positive_pct']:.1f}%")
    col4.metric("High Priority", summary['high_priority_count'],
                delta="Needs attention" if summary['high_priority_count'] > 0 else None,
                delta_color="inverse")
    
    st.markdown("---")
    
    # Key findings
    st.markdown("### Key Findings")
    st.markdown(f"""
- **Top Issue Category**: {summary['top_category'].replace('_', ' ').title()} ({summary['top_category_count']} mentions)
- **Sentiment Breakdown**: {summary['positive_pct']:.1f}% positive, {summary['negative_pct']:.1f}% negative
- **Items Requiring Attention**: {summary['high_priority_count']} high-priority items identified
    """)
    
    # High priority items
    if summary['high_priority_items']:
        st.markdown("### Top High-Priority Items")
        for i, item in enumerate(summary['high_priority_items'], 1):
            st.markdown(f"{i}. {item}")
    
    # Generate recommendations
    st.markdown("---")
    st.markdown("### Recommended Actions")
    
    with st.spinner("Generating recommendations..."):
        recommendations = generate_recommendations(summary, tagged_df, provider, model)
        st.markdown(recommendations)


def generate_recommendations(summary, tagged_df, provider, model):
    """Use LLM to generate actionable recommendations."""
    high_priority_text = "\n".join([f"- {item}" for item in summary['high_priority_items'][:10]])
    categories_text = "\n".join([f"- {k}: {v}" for k, v in summary['categories'].items()])
    
    prompt = f"""Based on this customer feedback analysis, provide 3-5 specific, actionable recommendations for the product team.

Analysis Summary:
- Total feedback items: {summary['total_items']}
- Top category: {summary['top_category']} ({summary['top_category_count']} mentions)
- Sentiment: {summary['positive_pct']:.1f}% positive, {summary['negative_pct']:.1f}% negative
- High priority items: {summary['high_priority_count']}

Category Breakdown:
{categories_text}

High Priority Items:
{high_priority_text if high_priority_text else "None identified"}

Provide specific, actionable recommendations. Format as a numbered list. Be concise but specific."""

    messages = [
        {"role": "system", "content": "You are a senior product manager providing actionable recommendations based on customer feedback analysis."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat(messages, provider=provider, model=model)
        return response
    except Exception as e:
        return f"Unable to generate recommendations: {str(e)}"


# ============================================================================
# Export Functions
# ============================================================================

def get_export_data(tagged_df, summary):
    """Prepare data for export."""
    # CSV export
    csv = tagged_df.to_csv(index=False)
    
    # JSON export
    json_data = tagged_df.to_json(orient='records', indent=2)
    
    # Markdown report
    report_md = f"""# Feedback Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Feedback Items | {summary['total_items']} |
| Positive Sentiment | {summary['positive_pct']:.1f}% |
| Negative Sentiment | {summary['negative_pct']:.1f}% |
| High Priority Items | {summary['high_priority_count']} |

## Category Breakdown

| Category | Count |
|----------|-------|
"""
    for cat, count in summary['categories'].items():
        report_md += f"| {cat.replace('_', ' ').title()} | {count} |\n"
    
    report_md += f"""
## High Priority Items

"""
    for i, item in enumerate(summary['high_priority_items'], 1):
        report_md += f"{i}. {item}\n"
    
    return csv, json_data, report_md


# ============================================================================
# Trend Analysis Functions
# ============================================================================

def display_trend_analysis(datasets, provider, model):
    """Display trend analysis comparing multiple datasets."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Install with: pip install plotly")
        return
    
    st.subheader("📈 Trend Analysis")
    
    # Combine all summaries
    trend_data = []
    for name, summary in datasets.items():
        for category, count in summary['categories'].items():
            trend_data.append({
                'Period': name,
                'Category': category.replace('_', ' ').title(),
                'Count': count
            })
    
    if not trend_data:
        st.warning("No data available for trend analysis")
        return
    
    trend_df = pd.DataFrame(trend_data)
    
    # Category trends
    st.markdown("### Category Trends Over Time")
    fig = px.line(
        trend_df, 
        x='Period', 
        y='Count', 
        color='Category',
        markers=True,
        title="Feedback Categories by Period"
    )
    fig.update_layout(xaxis_title="Time Period", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment trends
    sentiment_data = []
    for name, summary in datasets.items():
        sentiment_data.append({
            'Period': name,
            'Positive %': summary['positive_pct'],
            'Negative %': summary['negative_pct']
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    st.markdown("### Sentiment Trends Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sentiment_df['Period'], 
        y=sentiment_df['Positive %'],
        name='Positive',
        line=dict(color='#2ecc71'),
        mode='lines+markers'
    ))
    fig2.add_trace(go.Scatter(
        x=sentiment_df['Period'], 
        y=sentiment_df['Negative %'],
        name='Negative',
        line=dict(color='#e74c3c'),
        mode='lines+markers'
    ))
    fig2.update_layout(
        title="Sentiment Trends",
        xaxis_title="Time Period",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Delta indicators
    st.markdown("### Period-over-Period Changes")
    if len(datasets) >= 2:
        periods = list(datasets.keys())
        latest = datasets[periods[-1]]
        previous = datasets[periods[-2]]
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate deltas
        neg_delta = latest['negative_pct'] - previous['negative_pct']
        pos_delta = latest['positive_pct'] - previous['positive_pct']
        high_pri_delta = latest['high_priority_count'] - previous['high_priority_count']
        
        col1.metric(
            "Negative Sentiment", 
            f"{latest['negative_pct']:.1f}%",
            f"{neg_delta:+.1f}%",
            delta_color="inverse"
        )
        col2.metric(
            "Positive Sentiment",
            f"{latest['positive_pct']:.1f}%", 
            f"{pos_delta:+.1f}%"
        )
        col3.metric(
            "High Priority Items",
            latest['high_priority_count'],
            f"{high_pri_delta:+d}",
            delta_color="inverse"
        )
        
        # Summary
        st.markdown("---")
        st.markdown(f"""
**Key Changes ({periods[-2]} → {periods[-1]}):**
- Negative sentiment {"increased" if neg_delta > 0 else "decreased"} by {abs(neg_delta):.1f}%
- Positive sentiment {"increased" if pos_delta > 0 else "decreased"} by {abs(pos_delta):.1f}%
- High priority items {"increased" if high_pri_delta > 0 else "decreased"} by {abs(high_pri_delta)}
        """)
    else:
        st.info("Upload at least 2 datasets to see period-over-period changes")


# ============================================================================
# Authentication
# ============================================================================

def check_password():
    """Returns True if the user entered the correct password."""
    # Check if password is configured in secrets
    try:
        app_password = st.secrets.get("APP_PASSWORD", "")
    except:
        app_password = ""
    
    # If no password configured, allow access (for local development)
    if not app_password:
        return True
    
    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True
    
    # Show password form
    st.title("🔐 Feedback Analysis Tool")
    st.markdown("This app is password protected.")
    
    password = st.text_input("Enter password to continue", type="password")
    
    if password:
        if password == app_password:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    
    return False


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.set_page_config(
        page_title="Feedback Analysis Tool",
        page_icon="📊",
        layout="wide"
    )
    
    # Check password before showing app
    if not check_password():
        st.stop()
    
    st.title("📊 Feedback Analysis Tool")
    st.markdown("*AI-powered customer feedback analysis with actionable insights*")
    
    # Sidebar for LLM configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Provider selection
    provider_options = ["Local (Ollama)"]
    if GROQ_AVAILABLE and config.GROQ_API_KEY:
        provider_options.insert(0, "Groq (Cloud)")  # Prefer Groq if available
    
    provider_display = st.sidebar.selectbox(
        "LLM Provider",
        provider_options,
        help="Choose between local Ollama or cloud-based Groq"
    )
    
    provider = "ollama" if "Ollama" in provider_display else "groq"
    
    # Model selection
    if provider == "ollama":
        model = st.sidebar.text_input(
            "Ollama Model",
            value="llama3",
            help="Enter your local Ollama model name"
        )
        st.sidebar.info("🖥️ Using local Ollama. Ensure Ollama is running.")
    else:
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        model = st.sidebar.selectbox(
            "Groq Model",
            groq_models,
            help="Llama 3.3 70B recommended for best analysis quality"
        )
        st.sidebar.success("☁️ Using Groq cloud API")
    
    st.sidebar.markdown("---")
    
    # Analysis mode
    analysis_mode = st.sidebar.radio(
        "Mode",
        ["Single Analysis", "Trend Analysis"],
        help="Single: Analyze one dataset. Trend: Compare multiple datasets over time."
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "Data Type",
        ["Customer Feedback Data", "Product Support Tickets"]
    )
    
    # Main content
    st.markdown("---")
    
    if analysis_mode == "Trend Analysis":
        # Multi-file upload for trend analysis
        st.subheader("📈 Trend Analysis Mode")
        st.markdown("Upload multiple datasets to compare feedback trends over time.")
        
        uploaded_files = st.file_uploader(
            "📁 Upload multiple datasets (CSV or Excel)",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help="Upload 2 or more files representing different time periods"
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            # Let user label each file with a period name
            st.markdown("#### Label your time periods")
            period_names = {}
            text_columns = {}
            
            for i, file in enumerate(uploaded_files):
                col1, col2 = st.columns([1, 2])
                with col1:
                    period_names[file.name] = st.text_input(
                        f"Period label for {file.name}",
                        value=f"Period {i+1}",
                        key=f"period_{i}"
                    )
                with col2:
                    temp_df = read_file(file)
                    if temp_df is not None:
                        text_columns[file.name] = st.selectbox(
                            f"Text column for {file.name}",
                            temp_df.columns.tolist(),
                            key=f"col_{i}"
                        )
            
            if st.button("🚀 Run Trend Analysis", type="primary", use_container_width=True):
                all_summaries = {}
                
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    file.seek(0)  # Reset file pointer
                    df = read_file(file)
                    
                    if df is not None:
                        st.info(f"Analyzing {period_names[file.name]}...")
                        
                        # Tag the data
                        tagged_df = tag_feedback_rows(
                            df, text_columns[file.name], analysis_type,
                            provider, model,
                            progress_callback=lambda p: progress_bar.progress((i + p) / len(uploaded_files))
                        )
                        
                        # Generate summary
                        summary = generate_executive_summary(tagged_df)
                        all_summaries[period_names[file.name]] = summary
                
                progress_bar.progress(1.0)
                st.success("Trend analysis complete!")
                
                # Display trend analysis
                display_trend_analysis(all_summaries, provider, model)
        
        elif uploaded_files and len(uploaded_files) == 1:
            st.warning("Please upload at least 2 files for trend analysis")
        else:
            st.info("Upload 2 or more CSV/Excel files to begin trend analysis")
    
    else:
        # Single file upload
        uploaded_file = st.file_uploader(
            "📁 Upload your feedback data (CSV or Excel)",
            type=["csv", "xlsx"],
            help="Upload a file containing customer feedback or support tickets"
        )
    
        if uploaded_file:
            df = read_file(uploaded_file)
            
            if df is not None:
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(df)} rows")
                
                # Column selection
                text_column = st.selectbox(
                    "Select the column containing feedback text",
                    df.columns.tolist(),
                    help="Choose the column that contains the main feedback/ticket text"
                )
                
                # Large dataset options
                sample_size = None
                total_rows = len(df)
                
                if total_rows > 500:
                    st.markdown("---")
                    st.markdown("### ⚡ Large Dataset Options")
                    st.info(f"Your dataset has **{total_rows:,}** rows. For faster analysis, consider using sampling.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        use_sampling = st.checkbox(
                            "Enable sampling (recommended for large datasets)",
                            value=total_rows > 2000,
                            help="Analyze a representative sample instead of all rows"
                        )
                    
                    if use_sampling:
                        with col2:
                            sample_options = {
                                "500 rows (~2-3 min)": 500,
                                "1,000 rows (~4-5 min)": 1000,
                                "2,000 rows (~8-10 min)": 2000,
                                "5,000 rows (~20-25 min)": 5000,
                                "All rows (not recommended)": None
                            }
                            sample_choice = st.selectbox(
                                "Sample size",
                                list(sample_options.keys()),
                                index=1 if total_rows > 2000 else 0,
                                help="Larger samples = more accurate but slower"
                            )
                            sample_size = sample_options[sample_choice]
                        
                        if sample_size:
                            st.caption(f"Will analyze {sample_size:,} randomly selected rows ({sample_size/total_rows*100:.1f}% of data)")
                            estimated_batches = (sample_size + 14) // 15
                            st.caption(f"Estimated API calls: ~{estimated_batches} (batches of 15 rows each)")
                    else:
                        estimated_batches = (total_rows + 14) // 15
                        st.warning(f"⚠️ Processing all {total_rows:,} rows will require ~{estimated_batches} API calls and may take a long time.")
                
                st.markdown("---")
                
                # Analysis button
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    
                    # Check if already analyzed (in session state)
                    cache_key = f"{uploaded_file.name}_{text_column}_{analysis_type}_{sample_size}"
                    
                    if cache_key in st.session_state and st.session_state[cache_key] is not None:
                        tagged_df = st.session_state[cache_key]
                        st.info("Using cached results. Upload a new file to re-analyze.")
                    else:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        rows_to_process = sample_size if sample_size else total_rows
                        estimated_batches = (rows_to_process + 14) // 15
                        
                        def update_progress(pct):
                            progress_bar.progress(pct)
                            batch_num = int(pct * estimated_batches)
                            status_text.text(f"Analyzing... {int(pct * 100)}% complete (batch {batch_num}/{estimated_batches})")
                        
                        status_text.text(f"Starting analysis ({rows_to_process:,} rows in ~{estimated_batches} batches)...")
                        
                        try:
                            tagged_df = tag_feedback_rows(
                                df, text_column, analysis_type, 
                                provider, model, 
                                progress_callback=update_progress,
                                batch_size=15,
                                sample_size=sample_size
                            )
                            
                            # Cache results
                            st.session_state[cache_key] = tagged_df
                            
                            progress_bar.progress(1.0)
                            if sample_size and sample_size < total_rows:
                                status_text.text(f"Analysis complete! Analyzed {len(tagged_df):,} sampled rows.")
                            else:
                                status_text.text(f"Analysis complete! Analyzed {len(tagged_df):,} rows.")
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            return
                    
                    # Display results
                    st.markdown("---")
                    
                    # Generate summary
                    summary = generate_executive_summary(tagged_df)
                    
                    # Create tabs for different views
                    tab_names = [
                        "📊 Executive Summary",
                        "📈 Visual Dashboard", 
                        "📋 Tagged Data",
                        "📥 Export"
                    ]
                    if CREWAI_AVAILABLE:
                        tab_names.append("🔍 Root Cause Analysis")
                    
                    tabs = st.tabs(tab_names)
                    tab1, tab2, tab3, tab4 = tabs[:4]
                    tab5 = tabs[4] if len(tabs) > 4 else None
                    
                    with tab1:
                        display_executive_summary(summary, tagged_df, provider, model)
                    
                    with tab2:
                        display_visualizations(tagged_df)
                    
                    with tab3:
                        st.subheader("Tagged Feedback Data")
                        
                        # Filters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            filter_category = st.multiselect(
                                "Filter by Category",
                                tagged_df['_category'].unique().tolist(),
                                default=tagged_df['_category'].unique().tolist()
                            )
                        with col2:
                            filter_sentiment = st.multiselect(
                                "Filter by Sentiment",
                                tagged_df['_sentiment'].unique().tolist(),
                                default=tagged_df['_sentiment'].unique().tolist()
                            )
                        with col3:
                            filter_priority = st.multiselect(
                                "Filter by Priority",
                                tagged_df['_priority'].unique().tolist(),
                                default=tagged_df['_priority'].unique().tolist()
                            )
                        
                        # Apply filters
                        filtered_df = tagged_df[
                            (tagged_df['_category'].isin(filter_category)) &
                            (tagged_df['_sentiment'].isin(filter_sentiment)) &
                            (tagged_df['_priority'].isin(filter_priority))
                        ]
                        
                        st.dataframe(filtered_df, use_container_width=True)
                        st.caption(f"Showing {len(filtered_df)} of {len(tagged_df)} rows")
                    
                    with tab4:
                        st.subheader("Export Data")
                        
                        csv, json_data, report_md = get_export_data(tagged_df, summary)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                "📥 Download CSV",
                                csv,
                                "tagged_feedback.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.download_button(
                                "📥 Download JSON",
                                json_data,
                                "tagged_feedback.json",
                                "application/json",
                                use_container_width=True
                            )
                        
                        with col3:
                            st.download_button(
                                "📥 Download Report (MD)",
                                report_md,
                                "feedback_report.md",
                                "text/markdown",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                        st.markdown("### Report Preview")
                        st.markdown(report_md)
                    
                    # Root Cause Analysis Tab (if CrewAI available)
                    if tab5 is not None:
                        with tab5:
                            display_root_cause_analysis(
                                tagged_df, summary, provider, model, text_column
                            )


def display_root_cause_analysis(tagged_df, summary, provider, model, text_column):
    """Display the Root Cause Analysis tab with CrewAI agents."""
    st.subheader("🔍 Root Cause Analysis")
    st.markdown(
        "Use AI agents to correlate feedback with product documentation "
        "and release notes for deeper insights."
    )
    
    # Check if Groq is being used (required for CrewAI)
    if provider != "groq":
        st.warning(
            "⚠️ Root Cause Analysis requires Groq (cloud) provider. "
            "Please switch to Groq in the sidebar to use this feature."
        )
        return
    
    # Check for API key
    groq_api_key = config.GROQ_API_KEY
    if not groq_api_key:
        st.error("❌ Groq API key not found. Please configure it in secrets or .env file.")
        return
    
    st.markdown("---")
    
    # Document upload section
    st.markdown("### 📁 Upload Context Documents (Optional)")
    st.markdown(
        "Upload product documentation and release notes to enable deeper analysis. "
        "Supported formats: PDF, Markdown (.md), Text (.txt)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Product Documentation**")
        doc_files = st.file_uploader(
            "Upload product docs",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            key="doc_files",
            help="Upload product documentation, user guides, or feature specs"
        )
        if doc_files:
            st.success(f"✓ {len(doc_files)} doc(s) uploaded")
    
    with col2:
        st.markdown("**Release Notes / Changelog**")
        release_files = st.file_uploader(
            "Upload release notes",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            key="release_files",
            help="Upload release notes, changelogs, or version history"
        )
        if release_files:
            st.success(f"✓ {len(release_files)} release note(s) uploaded")
    
    st.markdown("---")
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    
    quick_mode = st.checkbox(
        "Quick Analysis Mode",
        value=True,
        help="Use single-agent analysis (faster, fewer API calls). Uncheck for full multi-agent analysis."
    )
    
    st.markdown("---")
    
    # Run analysis button
    if st.button("🚀 Run Root Cause Analysis", type="primary", use_container_width=True):
        run_root_cause_analysis(
            tagged_df, summary, provider, model, text_column,
            doc_files, release_files, quick_mode, groq_api_key
        )


def run_root_cause_analysis(
    tagged_df, summary, provider, model, text_column,
    doc_files, release_files, quick_mode, groq_api_key
):
    """Execute the root cause analysis with CrewAI agents."""
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # Process documents if uploaded
    docs_collection = None
    releases_collection = None
    
    with st.spinner("Processing documents..."):
        if doc_files:
            try:
                docs_collection = doc_processor.index_documents(
                    doc_files, "product_docs"
                )
                stats = doc_processor.get_document_stats(docs_collection)
                st.info(f"📚 Indexed {stats['total_chunks']} chunks from product docs")
            except Exception as e:
                st.warning(f"Could not process docs: {str(e)}")
        
        if release_files:
            try:
                releases_collection = doc_processor.index_documents(
                    release_files, "release_notes"
                )
                stats = doc_processor.get_document_stats(releases_collection)
                st.info(f"📋 Indexed {stats['total_chunks']} chunks from release notes")
            except Exception as e:
                st.warning(f"Could not process release notes: {str(e)}")
    
    # Prepare feedback context
    top_issues = tagged_df[tagged_df['_priority'] == 'high']['_summary'].tolist()[:10]
    if not top_issues:
        top_issues = tagged_df['_summary'].tolist()[:10]
    
    category_breakdown = tagged_df['_category'].value_counts().to_dict()
    
    # Create feedback summary text
    feedback_summary = f"""
Total feedback items: {summary['total_items']}
Positive sentiment: {summary['positive_pct']:.1f}%
Negative sentiment: {summary['negative_pct']:.1f}%
Neutral sentiment: {summary['neutral_pct']:.1f}%
High priority items: {summary['high_priority_count']}
"""
    
    # Initialize CrewAI
    crew = FeedbackAnalysisCrew(
        groq_api_key=groq_api_key,
        model=model,
        docs_collection=docs_collection,
        releases_collection=releases_collection
    )
    
    if quick_mode:
        # Quick single-agent analysis
        with st.spinner("🤖 Running quick analysis..."):
            try:
                sample_feedback = "\n".join([
                    f"- {row[text_column][:200]}" 
                    for _, row in tagged_df.head(20).iterrows()
                ])
                result = crew.quick_analyze(sample_feedback)
                
                st.markdown("### 📊 Quick Analysis Results")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                if "rate" in str(e).lower():
                    st.info("💡 Tip: Groq has rate limits. Wait a moment and try again.")
    else:
        # Full multi-agent analysis
        st.markdown("### 🤖 Agent Execution Progress")
        
        # Progress containers
        progress_container = st.container()
        agents = ["Feedback Analyst", "Documentation Specialist", "Release Analyst", "Insights Synthesizer"]
        agent_status = {agent: "pending" for agent in agents}
        
        # Create progress display
        with progress_container:
            progress_cols = st.columns(4)
            status_placeholders = {}
            for i, agent in enumerate(agents):
                with progress_cols[i]:
                    status_placeholders[agent] = st.empty()
                    status_placeholders[agent].markdown(f"⏳ **{agent}**\n\nPending...")
        
        def update_progress(progress):
            """Callback to update agent progress in UI."""
            agent_status[progress.agent_name] = progress.status
            if progress.agent_name in status_placeholders:
                if progress.status == "running":
                    status_placeholders[progress.agent_name].markdown(
                        f"🔄 **{progress.agent_name}**\n\n{progress.message or 'Running...'}"
                    )
                elif progress.status == "completed":
                    status_placeholders[progress.agent_name].markdown(
                        f"✅ **{progress.agent_name}**\n\nComplete"
                    )
                elif progress.status == "error":
                    status_placeholders[progress.agent_name].markdown(
                        f"❌ **{progress.agent_name}**\n\n{progress.message}"
                    )
        
        crew.progress_callback = update_progress
        
        with st.spinner("🤖 Running full agent analysis (this may take 1-2 minutes)..."):
            try:
                results = crew.analyze(
                    feedback_summary=feedback_summary,
                    top_issues=top_issues,
                    category_breakdown=category_breakdown
                )
                
                if results['success']:
                    st.markdown("---")
                    st.markdown("### 📋 Executive Report")
                    st.markdown(results['final_report'])
                    
                    # Download option
                    st.download_button(
                        "📥 Download Report",
                        results['final_report'],
                        "root_cause_analysis.md",
                        "text/markdown",
                        use_container_width=True
                    )
                else:
                    st.error(f"Analysis failed: {results['error']}")
                    if "rate" in str(results['error']).lower():
                        st.info("💡 Tip: Groq has rate limits (30 req/min). Wait a moment and try again, or use Quick Mode.")
                        
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                if "rate" in str(e).lower():
                    st.info("💡 Tip: Groq has rate limits. Try Quick Mode or wait a moment.")


if __name__ == "__main__":
    main()
