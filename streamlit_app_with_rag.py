import os
import sys

import mlflow
import streamlit as st
from langchain_openai import ChatOpenAI

# ローカルのPyTorchを無効化
# これをしないと `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_` みたいなエラーになる
# streamlitがソースコードのファイル変更を監視する時に起きるエラーらしく、検索対象を無効化してしまえば良い
sys.modules["torch.classes"] = None

# mlflow_rag.pyからRetrievalQA作成関数をインポート
from mlflow_rag import create_retrieval_qa

st.title("🦜🔗 RAG Quickstart App")

# Set the active experiment
mlflow.set_experiment("Legal RAG")


# RetrievalQAチェーンを初期化（キャッシュして一度だけ実行する）
@st.cache_resource
def initialize_retrieval_qa():
    """RetrievalQAチェーンを初期化する"""
    return create_retrieval_qa()


# RetrievalQAチェーンを取得
retrieval_qa = initialize_retrieval_qa()


@mlflow.trace(
    name="rag_llm_call",
    attributes={"model": "gemma-3-12b", "source": "local", "type": "RAG"},
)
def generate_response_with_rag(input_text):
    # RAGを使用してレスポンスを生成
    # RetrievalQAチェーンを使用して回答を生成
    result = retrieval_qa.invoke({"query": input_text})
    output = result["result"]

    # 参照されたドキュメントも表示（オプション）
    if "source_documents" in result:
        with st.expander("参照されたドキュメント"):
            for i, doc in enumerate(result["source_documents"]):
                st.write(f"**ソース {i+1}:**")
                st.write(
                    doc.page_content[:500] + "..."
                    if len(doc.page_content) > 500
                    else doc.page_content
                )

    st.info(output)
    return output


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What does the document say about trespassers?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response_with_rag(text)
