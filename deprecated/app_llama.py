import streamlit as st
import pandas as pd
import ollama


def ollama_function(messages, model='llama3'):
    llama3 = ollama.chat(model=model, messages=messages)
    return llama3['message']['content']


def read_file(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # Check for missing headers or empty rows at the start
    if df.iloc[0].isnull().all():
        df.columns = df.iloc[1]  # Use the second row as header
        df = df[2:]  # Skip the first two rows
    elif df.columns.isnull().all():
        df.columns = df.iloc[0]  # Use the first row as header
        df = df[1:]  # Skip the first row

    df.reset_index(drop=True, inplace=True)
    return df


def analyze_data(data, selected_columns, analysis_type):
    combined_text = " ".join(data[selected_columns].astype(str).apply(lambda x: ' '.join(x), axis=1))
    if analysis_type == "Customer Feedback Data":
        prompt = (
            f"Analyze the following customer feedback and extract the following information:\n"
            f"1. Top feature requests from the customers\n"
            f"2. Most loved features based on user appreciation\n"
            f"3. Main complaints in terms of what seems to worry the user, or is a painful experience\n"
            f"4. Competitor mentions, such as any other platform that they could be praising\n\n"
            f"Customer feedback:\n{combined_text}"
        )
    else:
        prompt = (
            f"Analyze the following product support tickets and extract the following information:\n"
            f"1. Critical issues keeping in view what could affect business, or something that is causing great user distress\n"
            f"2. Looming issues, which is issues which are a level below critical, but are important\n"
            f"3. Potential fixes for issues as to what can be done to fix the issues; suggestions\n\n"
            f"Support tickets:\n{combined_text}"
        )
    
    messages = [
        {"role": "system", "content": "You are an expert data analyst."},
        {"role": "user", "content": prompt}
    ]

    response = ollama_function(messages)
    if response:
        return response
    return None


def extract_insights(analysis_text, analysis_type):
    """Extract insights from the analysis text."""
    if analysis_type == "Customer Feedback Data":
        return {
            "top_feature_requests": analysis_text,
            "most_loved_features": analysis_text,
            "main_complaints": analysis_text,
            "competitor_mentions": analysis_text,
        }
    else:
        return {
            "critical_issues": analysis_text,
            "looming_issues": analysis_text
        }


# Streamlit UI
st.title("Feedback and Support Ticket Analysis Tool")

file_type = st.radio("Select file type", ["Customer Feedback Data", "Product Support Tickets"])

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = read_file(uploaded_file)

    st.write("Dataset Preview:", df.head())

    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to include in analysis", all_columns, default=all_columns)

    if st.button("Run Analysis"):
        analysis_text = analyze_data(df, selected_columns, file_type)
        if analysis_text:
            analysis_result = extract_insights(analysis_text, file_type)
            if file_type == "Customer Feedback Data":
                st.subheader("Top Feature Requests")
                st.write(analysis_result.get("top_feature_requests", "No data"))

                st.subheader("Most Loved Features")
                st.write(analysis_result.get("most_loved_features", "No data"))

                st.subheader("Main Complaints")
                st.write(analysis_result.get("main_complaints", "No data"))

                st.subheader("Competitor Mentions")
                st.write(analysis_result.get("competitor_mentions", "No data"))
            elif file_type == "Product Support Tickets":
                st.subheader("Critical Issues")
                st.write(analysis_result.get("critical_issues", "No data"))

                st.subheader("Looming Issues")
                st.write(analysis_result.get("looming_issues", "No data"))
