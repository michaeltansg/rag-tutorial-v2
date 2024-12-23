import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_embedding_function():
    # Create the OpenAIEmbeddings instance with your API key
    embeddings = OpenAIEmbeddings(
        model=os.getenv('EMBEDDING_MODEL', "text-embedding-ada-002")
    )
    
    return embeddings