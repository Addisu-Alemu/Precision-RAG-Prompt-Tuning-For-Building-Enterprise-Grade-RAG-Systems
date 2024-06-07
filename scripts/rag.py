# from langchain_community.document_loaders import DirectoryLoader
# DATA_PATH = "../data"
# loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
# documents = loader.load()
# documents

# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load the .env file

# OPENAI_API = os.getenv("OPENAI_API")  # Get the API key from the environment

# # Use the api_key variable in your code
# print(OPENAI_API)

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
# chunks = text_splitter.split_documents(documents)

# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma

# import openai 
# Chroma_path = "chroma"

# db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=Chroma_path)
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# import pandas as pd
# import openai 
# embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API"))
# all_embeddings = []
# for chunk in chunks:
#     embedding = embeddings.embed_query(chunk.page_content)
#     all_embeddings.append({
#         "id": chunk.metadata.get("id", None),  # Assuming you have an "id" in your metadata
#         "text": chunk.page_content,
#         "embedding": embedding,
#     })

# # 4. Create a DataFrame from the Embeddings
# df = pd.DataFrame(all_embeddings)

# # 5. Save the DataFrame to a CSV file
# df.to_csv("embeddings.csv", index=False)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd

def load_environment():
    """Loads environment variables from a .env file."""
    load_dotenv()  # Load the .env file
    return os.getenv("OPENAI_API")

def load_documents(data_path, glob_pattern):
    """Loads documents from a directory."""
    loader = DirectoryLoader(data_path, glob=glob_pattern)
    return loader.load()

def split_documents(documents, chunk_size=300, chunk_overlap=100):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def create_embeddings(chunks, api_key):
    """Generates embeddings for each document chunk."""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    all_embeddings = []
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)
        all_embeddings.append({
            "id": chunk.metadata.get("id", None),
            "text": chunk.page_content,
            "embedding": embedding,
        })
    return all_embeddings

def save_embeddings_to_csv(embeddings, filename="embeddings.csv"):
    """Saves embeddings to a CSV file."""
    df = pd.DataFrame(embeddings)
    df.to_csv(filename, index=False)

def main():
    """Main function to run the embedding process."""
    # Load environment variables
    OPENAI_API = load_environment()

    # Load and split documents
    DATA_PATH = "../data"
    documents = load_documents(DATA_PATH, "*.pdf")
    chunks = split_documents(documents)

    # Generate embeddings
    embeddings = create_embeddings(chunks, OPENAI_API)

    # Save embeddings to CSV
    save_embeddings_to_csv(embeddings)

if __name__ == "__main__":
    main()