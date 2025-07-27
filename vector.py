from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Read the cleaned CSV
df = pd.read_csv("Data.csv")

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define location for the vector store
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Initialize vector store with embeddings
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# If the vector store is new, create documents
if add_documents:
    documents = []
    ids = []
    
    # Loop through the CSV and create documents
    for i, row in df.iterrows():
        # Fixing extra commas by making sure the answer is clean
        document = Document(
            page_content=row["question"],
            metadata={"Answer": row["answer"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    # Add documents to the vector store
    vector_store.add_documents(documents=documents, ids=ids)

# Create a retriever from the vector store to fetch relevant documents based on the question
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
