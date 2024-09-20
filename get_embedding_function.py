from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    # Use LLaMA 3 for embeddings
    return OllamaEmbeddings(model="llama3")
