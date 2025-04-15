"""Retrieval tool for Agroz Farm and E-commerce information."""

import os
import glob
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from bs4 import BeautifulSoup
import re

# Define paths
CURRENT_DIR = Path(__file__).parent
DOCS_DIR = CURRENT_DIR / "docs"
CHROMA_DB_DIR = CURRENT_DIR / "chroma_db_gpt4all"


def simple_extractor(html: str) -> str:
    """Extract text from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def get_documents(path: str):
    """Load documents from a Word file."""
    return UnstructuredWordDocumentLoader(path).load()


def load_website_doc():
    """Load documents from Agroz website."""
    return RecursiveUrlLoader(
        url="https://agroz.co/",
        max_depth=4,
        extractor=simple_extractor
    ).load()


def ingest_docs():
    """Ingest documents into vector database."""
    print("Starting document ingestion process...")
    
    # Initialize text splitters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_splitter_website = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # Initialize embeddings model
    print("Initializing GPT4AllEmbeddings model...")
    embedding = GPT4AllEmbeddings()
    
    # Load documents from website
    print("Loading documents from Agroz website...")
    try:
        # docs_from_website = load_website_doc()
        docs_from_website = WebBaseLoader(
            web_path="https://agroz.co/"
        ).load()

        print(f"Loaded {len(docs_from_website)} documents from website")
        
        # Print sample of website content after extraction
        if docs_from_website:
            print("\nSample website content after extraction:")
            sample_content = docs_from_website[0].page_content[:500] + "..." if len(docs_from_website[0].page_content) > 500 else docs_from_website[0].page_content
            print(sample_content)
            print("\n")
        
        docs_web_transformed = text_splitter_website.split_documents(docs_from_website)
        print(f"Split website documents into {len(docs_web_transformed)} chunks")
    except Exception as e:
        print(f"Error loading website documents: {e}")
        docs_web_transformed = []
    
    
    # Load documents from Word files
    print("\nLoading documents from Word files...")
    docx_files = glob.glob(str(DOCS_DIR / '*.docx'))
    documents = []
    for file in docx_files:
        try:
            print(f"Processing document: {os.path.basename(file)}")
            documents.append(get_documents(file)[0])
        except Exception as e:
            print(f"Error processing document {os.path.basename(file)}: {e}")
    
    print(f"Loaded {len(documents)} documents from Word files")
    
    # Split documents
    print("Splitting documents...")
    docs_transformed = text_splitter.split_documents(documents)
    print(f"Split Word documents into {len(docs_transformed)} chunks")
    
    # Filter out documents with very short content
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]
    docs_web_transformed = [doc for doc in docs_web_transformed if len(doc.page_content) > 10]
    
    # Combine all documents
    all_docs = docs_transformed + docs_web_transformed
    print(f"Total document chunks after filtering: {len(all_docs)}")
    
    # Add metadata if missing
    for doc in all_docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
    
    # Create vector store
    print("Creating Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=filter_complex_metadata(all_docs), 
        embedding=embedding,
        persist_directory=str(CHROMA_DB_DIR)
    )
    
    print("Document ingestion complete!")
    return vectorstore


def ensure_db_exists():
    """Ensure the vector database exists, creating it if necessary."""
    if not CHROMA_DB_DIR.exists():
        print("Vector database does not exist. Creating it now...")
        ingest_docs()
    else:
        print("Vector database already exists.")


def get_retriever():
    """Get a retriever from the vector store."""
    # Load existing vector store
    embedding = GPT4AllEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding
    )
    
    # Create retriever
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 4, 'score_threshold': 0.3}
    )


# Ensure the database exists when this module is imported
print("Initializing Agroz retrieval tool...")
ensure_db_exists()


async def retrieve_agroz_info(query: str, max_results: Optional[int] = 5) -> str:
    """Retrieve information about Agroz Farm and E-commerce based on the query.
    
    Use this tool when you need to find specific information about:
    - Agroz Farm operations, practices, and procedures
    - Agroz E-commerce website products and services
    - Growing information for various crops (tomatoes, basil, kale, etc.)
    - Standard operating procedures for Agroz
    - Any agricultural or farming-related questions specific to Agroz
    
    This tool searches through Agroz documentation and website content to provide
    accurate and relevant information about Agroz's agricultural practices,
    products, services, and farming techniques.
    
    Args:
        query: The search query about Agroz Farm or E-commerce.
        max_results: Maximum number of results to return (default: 5).
        
    Returns:
        A string containing the relevant information from Agroz documentation.
    """
    # Ensure max_results is valid
    if max_results is None or max_results < 1:
        max_results = 5
    
    try:
        # Get retriever
        retriever = get_retriever()
        
        # Retrieve documents
        docs = retriever.get_relevant_documents(query)
        
        # Limit results
        docs = docs[:max_results]
        
        if not docs:
            return f"No information found for query: '{query}'"
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown source")
            content = doc.page_content.strip()
            results.append(f"Result {i}:\nSource: {source}\n{content}\n")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error retrieving information: {str(e)}"
