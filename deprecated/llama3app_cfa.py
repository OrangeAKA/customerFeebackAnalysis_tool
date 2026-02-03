import streamlit as st
import pandas as pd
import ollama


def ollama_function(messages, model='llama3'):
    llama3 = ollama.chat(model=model, messages=messages)
    return llama3['message']['content']


def read_file(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error('Unsupported format; please upload files with extension .csv or .xlsx')
        return None

    if df.iloc[0].isnull().all():
        df.columns = df.iloc[1]
        df = df[2:]
    elif df.columns.isnull().all():
        df.columns = df.iloc[0]
        df = df[1:]

    df.reset_index(drop=True, inplace=True)
    return df


def analyze_data(data, selected_columns, analysis_type):
    combined_text = " ".join(data[selected_columns].astype(str).apply(lambda x: ' '.join(x), axis=1))
    if analysis_type == "Customer Feedback Data":
        prompt = (
            f"Analyze the following customer feedback data and extract the following information:\n"
            f"1. Top feature requests\n"
            f"2. Most loved features\n"
            f"3. Main complaints\n"
            f"4. Competitor mentions\n\n"
            f"Customer feedback:\n{combined_text}"
        )
    else:
        prompt = (
            f"Analyze the following product support tickets and extract the following information:\n"
            f"1. Critical issues, based on how frequent a certain issue is observed, and the deemed criticality of it from a functionality and severity standpoint\n"
            f"2. Looming issues, which are not immediately critical, but could be serious and have to be taken up for being addressed with a fix or relevant solution at the earliest\n"
            f"3. Potential fixes for each of the Critical and looming issues\n"
            f"Support tickets:\n{combined_text}"
        )

    messages = [
        {"role": "system", "content": "You are an expert Product Analyst with good analytical skills."},
        {"role": "user", "content": prompt}
    ]

    response = ollama_function(messages)
    if response:
        analysis = response.strip()
        return analysis
    return None


def extract_section(text, section_title):
    lines = text.split('\n')
    start_index = next((i for i, line in enumerate(lines) if section_title.lower() in line.lower()), None)
    if start_index is None:
        return "No data found"
    
    end_index = next((i for i, line in enumerate(lines[start_index + 1:], start=start_index + 1) if line.strip() == ""), len(lines))
    return "\n".join(lines[start_index + 1:end_index]).strip()


def display_analysis(analysis_text, analysis_type):
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
st.title("Customer Feedback Analysis Tool")

file_type = st.radio("Select file type", ["Customer Feedback Data", "Product Support Tickets"])

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
        st.write("Dataset Preview:", df.head())

        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns to include in analysis", all_columns, default=all_columns)

        if st.button("Run Analysis"):
            st.info("Running analysis...")
            analysis_text = analyze_data(df, selected_columns, file_type)
            st.info("Analysis completed")
            if analysis_text:
                display_analysis(analysis_text, file_type)


# Note: Headers are metadata, and not counted towards the actual data rows itself.
# Thus, gotta be careful when using iloc to subsequently re-index
