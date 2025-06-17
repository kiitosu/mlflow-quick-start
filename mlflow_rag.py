import os
import shutil  # for deleting temporary files
import tempfile

import mlflow
import requests
from bs4 import BeautifulSoup  # HTMLのパースに使う
from langchain.chains import RetrievalQA  # データベースからの検索に使う
from langchain.document_loaders import TextLoader  # テキストファイルを読み込む
from langchain.text_splitter import CharacterTextSplitter  # テキストを分割する
from langchain.vectorstores import FAISS  # ベクトルデータベースを作る
from langchain_openai import OpenAI, OpenAIEmbeddings  # OpenAI APIを使う

assert (
    "OPENAI_API_KEY" in os.environ
), "Please set the OPENAI_API_KEY environment variable."


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

    # Generate embeddings for each document chunk
    embedding_generator = OpenAIEmbeddings()
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

mlflow.set_experiment("Legal RAG")

retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=vector_db.as_retriever())


# Log the retrievalQA chain
def load_retriever(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True,  # This is required to load the index from MLflow
    )
    return vectorstore.as_retriever()


with mlflow.start_run() as run:
    model_info = mlflow.langchain.log_model(
        retrievalQA,
        name="retrieval_qa",
        loader_fn=load_retriever,
        persist_dir=persist_dir,
    )
