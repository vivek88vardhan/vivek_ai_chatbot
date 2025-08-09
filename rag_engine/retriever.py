from langchain.vectorstores import Weaviate
from langchain.embeddings import SentenceTransformerEmbeddings

def retrieve_docs(query):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Weaviate("http://localhost:8080", embedding=embeddings)
    results = db.similarity_search(query)
    return [doc.page_content for doc in results]
