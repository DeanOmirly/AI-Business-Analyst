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
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# LangChain RAG tools
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# UI Config
st.set_page_config(page_title="📊 AI Business Analyst + Auto-RAG", layout="wide")
st.title("💼 AI-Powered Business Data Analyst")

# File Upload + Input
uploaded_file = st.file_uploader("📥 Upload your business dataset (CSV)", type="csv")
question = st.text_input("💬 What business insight or analysis are you looking for?")

# Optional checkboxes
show_rag_context = st.checkbox("Show RAG context")
require_plot = st.checkbox("Always include a plot in the analysis")


# LLM prompt
template = """
You are a Python data analyst with expertise in Business Analysis.
Use ONLY the user-provided datasetwhich is called df.
Here is expert context of what it means to be a business analyst. Please use this context to support your response if needed or relevant, keep it concise and choose the two most similar text chunks, no more:

{context_text}

A user uploaded this dataset sample:

{data_sample}

User question: "{question}"

   - Do **not** use `df.head()` or subsets of the data unless asked. Use **at least 100 rows** of the dataset (`df`) for analysis.

With the provided data (df):
1. Answer the User Question to the best of your ability and anything it asks for. Make sure you accoutn for how visualizations will look if using all observations, so know when to use a lot and when to choose select groups.
"""

prompt = PromptTemplate(
    input_variables=["data_sample", "question", "context_text", "var_info"],
    template=template,
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_chain = prompt | llm

# Load the retriever from the FAISS vector store (auto-loaded index)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", k=2)




# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='cp1252')
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write(f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Show all columns"):
        st.write("Columns:", list(df.columns))

    if st.checkbox("Show summary statistics"):
        st.write(df.describe(include="all"))

    # Save to session
    st.session_state.df = df

if question and st.button("Generate Analysis"):
    sample_size = min(max(len(df), 50), 100)
    data_sample = df.head(sample_size).to_csv(index=False)

    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    col_types = df.dtypes.apply(str).to_dict()
    categorical_vars = [col for col, dtype in col_types.items() if dtype == "object"]
    numeric_vars = [
        col for col, dtype in col_types.items() if dtype in ["int64", "float64"]
    ]
    var_info = f"""
Your dataset contains:
- Categorical: {", ".join(categorical_vars) or "None"}
- Numeric: {", ".join(numeric_vars) or "None"}
"""

    modified_question = question
    if require_plot:
        modified_question += " Please include at least one appropriate plot."

    result = llm_chain.invoke(
        {
            "data_sample": data_sample,
            "question": modified_question,
            "context_text": context_text,
            "var_info": var_info,
        }
    )

    output_text = result.content if hasattr(result, "content") else result
    st.session_state.generated_code = output_text
    st.session_state.generated_response = output_text
    st.session_state.context_text = context_text
    st.session_state.question = question

# Show result and RAG context
if "generated_response" in st.session_state:
    st.markdown("### Suggested Analysis & Code")
    st.markdown(st.session_state.generated_response)

if show_rag_context and "context_text" in st.session_state:
    st.markdown("### Retrieved RAG Context")
    st.markdown(st.session_state.context_text)

# Run the generated code
if "generated_code" in st.session_state and st.button("Run the code"):
    code = st.session_state.generated_code
    code_match = re.search(r"```python(.*?)```", code, re.DOTALL)
    code_to_run = code_match.group(1).strip() if code_match else code

    required_packages = ["statsmodels", "seaborn", "matplotlib", "scipy"]
    missing = [
        pkg for pkg in required_packages if __import__(pkg, None, None, [], 0) is None
    ]

    if missing:
        st.error(f"Missing packages: {', '.join(missing)}")
    else:
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                exec(
                    code_to_run,
                    {
                        "df": st.session_state.df,
                        "pd": pd,
                        "np": np,
                        "plt": plt,
                        "sns": sns,
                        "stats": stats,
                        "sm": sm,
                        "smf": smf,
                        "st": st,
                    },
                )
            output = f.getvalue()
            st.markdown("### Code Output")
            st.text(output)

            import matplotlib.pyplot as mpl_pyplot

            figs = [mpl_pyplot.figure(n) for n in mpl_pyplot.get_fignums()]
            for fig in figs:
                st.pyplot(fig)
                mpl_pyplot.close("all")

            st.session_state.code_output = output
        except Exception as e:
            st.error(f"🚫 Error running code: {e}")

# Summarize output
if "code_output" in st.session_state and st.button("Summarize Output"):
    question = st.session_state.get("question")
    if not question:
        st.warning("Please generate analysis first.")
    else:
        context_text = st.session_state.get("context_text", "")
        summary_prompt = PromptTemplate(
            input_variables=["output", "context_text"],
            template="""
You are a data assistant with expertise in Business Analysis.
Use this expert context if relevant:

{context_text}

Now summarize this output:

{output}
""",
        )
        summary_chain = summary_prompt | llm
        with st.spinner("Summarizing..."):
            summary_result = summary_chain.invoke(
                {
                    "output": st.session_state.code_output,
                    "context_text": context_text,
                }
            )
        summary_text = (
            summary_result.content
            if hasattr(summary_result, "content")
            else summary_result
        )
        st.markdown("### Summary of Statistical Output")
        st.markdown(summary_text)
