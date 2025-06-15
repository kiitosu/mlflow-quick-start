import mlflow
import streamlit as st
from langchain_openai import ChatOpenAI

st.title("🦜🔗 Quickstart App")

# Set the active experiment
mlflow.set_experiment("my-genai-experiment")


@mlflow.trace(name="llm_call", attributes={"model": "gemma-3-12b", "source": "local"})
def generate_response(input_text):
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0.7,
        model_name="google/gemma-3-12b",
    )
    output = llm.invoke(input_text).content
    st.info(output)
    return output


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
