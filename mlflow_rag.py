import os
import tempfile
import warnings

import mlflow
import requests
from bs4 import BeautifulSoup  # HTMLのパースに使う
from langchain._api import LangChainDeprecationWarning
from langchain.chains import RetrievalQA  # データベースからの検索に使う
from langchain.text_splitter import CharacterTextSplitter  # テキストを分割する
from langchain_community.document_loaders import (
    TextLoader,  # テキストファイルを読み込む
)
from langchain_community.vectorstores import FAISS  # ベクトルデータベースを作る
from langchain_huggingface import HuggingFaceEmbeddings  # ローカル埋め込み用
from langchain_openai import ChatOpenAI  # LM Studio用

# LangChainの非推奨警告を抑制
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# HuggingFaceトークナイザーの並列処理警告を抑制
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fetch_federal_document(url, div_class):  # noqa: D417
    """
    Scrapes the transcript of the Act Establishing Yellowstone National Park from the given URL.

    Args:
    url (str): URL of the webpage to scrape.

    Returns:
    str: The transcript text of the Act.
    """
    # Sending a request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parsing the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Finding the transcript section by its HTML structure
        transcript_section = soup.find("div", class_=div_class)
        if transcript_section:
            transcript_text = transcript_section.get_text(separator="\n", strip=True)
            return transcript_text
        else:
            return "Transcript section not found."
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"


def fetch_and_save_documents(url_list, doc_path):
    """
    Fetches documents from given URLs and saves them to a specified file path.

    Args:
        url_list (list): List of URLs to fetch documents from.
        doc_path (str): Path to the file where documents will be saved.
    """
    for url in url_list:
        document = fetch_federal_document(url, "col-sm-9")
        with open(doc_path, "a") as file:
            file.write(document)


def create_faiss_database(
    document_path, database_save_directory, chunk_size=500, chunk_overlap=10
):
    """
    Creates and saves a FAISS database using documents from the specified file.

    Args:
        document_path (str): Path to the file containing documents.
        database_save_directory (str): Directory where the FAISS database will be saved.
        chunk_size (int, optional): Size of each document chunk. Default is 500.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Default is 10.

    Returns:
        FAISS database instance.
    """
    # Load documents from the specified file
    document_loader = TextLoader(document_path)
    raw_documents = document_loader.load()

    # Split documents into smaller chunks with specified size and overlap
    document_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    document_chunks = document_splitter.split_documents(raw_documents)

    # HuggingFace埋め込みモデルを使用
    embedding_generator = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # CPUを使用（GPUがある場合は'cuda'）
    )
    faiss_database = FAISS.from_documents(document_chunks, embedding_generator)

    # Save the FAISS database to the specified directory
    faiss_database.save_local(database_save_directory)

    return faiss_database


temporary_directory = tempfile.mkdtemp()

doc_path = os.path.join(temporary_directory, "docs.txt")
persist_dir = os.path.join(temporary_directory, "faiss_index")

url_listings = [
    "https://www.archives.gov/milestone-documents/act-establishing-yellowstone-national-park#transcript",
    "https://www.archives.gov/milestone-documents/sherman-anti-trust-act#transcript",
]

fetch_and_save_documents(url_listings, doc_path)

vector_db = create_faiss_database(doc_path, persist_dir)

# MLflowサーバーのURIを設定
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Legal RAG")

# LM StudioのGemmaを使用
local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key=None,  # LM Studioなのでダミー値
    temperature=0.7,
    model="gemma",  # LM Studioで起動しているモデル名
)

retrievalQA = RetrievalQA.from_llm(llm=local_llm, retriever=vector_db.as_retriever())


# Log the retrievalQA chain
def load_retriever(persist_directory):
    # 同じHuggingFace埋め込みモデルを使用
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True,  # This is required to load the index from MLflow
    )
    return vectorstore.as_retriever()


with mlflow.start_run() as run:
    # 依存関係を明示的に指定
    # mlflow.utils.environment: Encountered an unexpected error while inferring pip requirements の警告を抑制するため
    pip_requirements = [
        "langchain==0.3.25",
        "langchain-community",
        "langchain-openai",
        "langchain-huggingface",
        "sentence-transformers",
        "faiss-cpu",
        "beautifulsoup4",
        "requests",
        "pydantic==2.11.7",
        "cloudpickle==3.1.1",
    ]

    model_info = mlflow.langchain.log_model(
        retrievalQA,
        name="retrieval_qa",
        loader_fn=load_retriever,
        persist_dir=persist_dir,
        pip_requirements=pip_requirements,  # 明示的に指定
    )


# チャットアプリから使えるように一連の作業を関数化する
def create_retrieval_qa():
    """
    RetrievalQAチェーンを作成して返す

    Returns:
        RetrievalQA: 設定済みのRetrievalQAチェーン
    """
    # ドキュメントの取得と保存
    temporary_directory = tempfile.mkdtemp()

    local_doc_path = os.path.join(temporary_directory, "docs.txt")

    fetch_and_save_documents(url_listings, local_doc_path)

    # ドキュメントの読み込み
    loader = TextLoader(local_doc_path, encoding="utf-8")
    documents = loader.load()

    # テキストの分割
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 埋め込みモデルの設定
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ベクトルストアの作成
    vectorstore = FAISS.from_documents(docs, embeddings)

    # LLMの設定
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key=None,
        temperature=0.7,
        model_name="google/gemma-3-12b",
    )

    # RetrievalQAチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    return qa_chain
