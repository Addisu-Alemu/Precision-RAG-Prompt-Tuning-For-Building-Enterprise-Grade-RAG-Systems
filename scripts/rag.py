from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import argparse

def load_and_split_pdf(pdf_path):
    """Loads and splits a PDF document into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def create_chroma_db(chunks, chroma_path):
    """Creates a Chroma database from the document chunks."""
    load_dotenv()
    embedding_function = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API"))
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=chroma_path)
    return db

def query_chroma_db(db, query_text):
    """Queries the Chroma database for relevant results."""
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return None, None

    context_text = "\n\n------------\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return context_text, sources

def get_answer(context_text, query_text):
    """Generates an answer to the query based on the context."""
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(prompt)
    model = ChatOpenAI(api_key=os.getenv("OPENAI_API"))
    response_text = model.predict(prompt)
    return response_text

def main():
    """Main function to process the PDF and answer the query."""
    parser = argparse.ArgumentParser(description="Process a PDF and answer a query.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument("query_text", help="The query to answer.")
    args = parser.parse_args()

    chunks = load_and_split_pdf(args.pdf_path)
    chroma_path = "chroma"
    db = create_chroma_db(chunks, chroma_path)

    context_text, sources = query_chroma_db(db, args.query_text)
    if context_text is None:
        return

    response_text = get_answer(context_text, args.query_text)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()