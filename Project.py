# C:\Users\omirl\OneDrive\Desktop\QAC387\Assignment 3\ai-data-analysis-assistant
# .\venv\Scripts\Activate.ps1


# Project.py
import os
import io
import contextlib
import re
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# === Load environment variables from .env file ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Streamlit UI ===
st.set_page_config(page_title="📊 Business Analyst Assistant", layout="wide")
st.title("💼 AI-Powered Business Data Analyst")

# === File Upload and Question Input ===
uploaded_file = st.file_uploader("📥 Upload your business dataset (CSV)", type="csv")
question = st.text_input("💬 What business insight or analysis are you looking for?")

# === Initialize LLM Chain ===
template = """
You are an AI business analyst. A user has uploaded the following business dataset sample:

{data_sample}

The user asked: "{question}"

Determine the most appropriate business analysis (e.g., financial ratio analysis, customer segmentation, forecasting),
explain the method, and generate Python code using the uploaded dataset, which is already loaded in a variable 
called `df`.

Include descriptive statistics, and when appropriate, provide recommendations for visualizations and business decisions. 
Use `print()` or `st.write()` to interpret your results. Round all numeric output to 2 decimal places.

Only include the explanation and the code in your response.
"""
prompt = PromptTemplate(input_variables=["data_sample", "question"], template=template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_chain = prompt | llm

# === Main logic ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fix Arrow serialization issues
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if st.checkbox("🔍 Show column names"):
        st.write(list(df.columns))

    if st.checkbox("📄 Show full dataset"):
        st.dataframe(df)

    if st.checkbox("📈 Show summary statistics"):
        st.write(df.describe(include="all"))

    if question and st.button("Generate Business Insight"):
        with st.spinner("Analyzing your data..."):
            data_sample = df.head(10).to_csv(index=False)
            result = llm_chain.invoke({"data_sample": data_sample, "question": question})

        st.markdown("### 🧠 Suggested Analysis & Code")
        output_text = result.content if hasattr(result, "content") else result
        st.markdown(output_text if output_text else "*No response received from LLM*")
        st.session_state.generated_code = output_text

# === Optional: Execute Suggested Code ===
if "generated_code" in st.session_state:
    if st.button("▶️ Run Suggested Code"):
        result = st.session_state.generated_code

        code_match = re.search(r"```python(.*?)```", result, re.DOTALL)
        code_to_run = code_match.group(1).strip() if code_match else result

        st.markdown("### 📤 Code Output")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                exec(code_to_run, {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "st": st})
                output = f.getvalue()
                st.text(output)
            except Exception as e:
                st.error(f"❌ Error running code: {e}")
